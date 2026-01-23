import warnings
from typing import Callable, List, Dict

import casadi as ca
import openap.casadi as oc
from openap.extra.aero import fpm, ft, kts

import numpy as np
import openap
import pandas as pd

try:
    from . import tools
except Exception:
    RuntimeWarning("cfgrib and sklearn are required for wind integration")


class Base:
    def __init__(
        self,
        scenarios: List[Dict],
        dT: float = 0.0,
        use_synonym: bool = False,
    ):
        """OpenAP trajectory optimizer.

        Args:
            actype (str): ICAO aircraft type code
            origin (Union[str, tuple]): ICAO or IATA code of airport, or tuple (lat, lon)
            destination (Union[str, tuple]): ICAO or IATA code of airport, or tuple (lat, lon)
            m0 (float, optional): Takeoff mass factor. Defaults to 0.8 (of MTOW).
            dT (float, optional): Temperature shift from standard ISA. Default = 0
        """
        self.dT = dT
        self.use_synonym = use_synonym
        self.scenarios = scenarios
        self.lat0 = None
        self.lon0 = None

        for sc in self.scenarios:
            actype = sc["actype"]
            if isinstance(sc["origin"], str):
                ap1 = openap.nav.airport(sc["origin"])
                sc["lat1"], sc["lon1"] = ap1["lat"], ap1["lon"]
            else:
                sc["lat1"], sc["lon1"] = sc["origin"]

            if isinstance(sc["destination"], str):
                ap1 = openap.nav.airport(sc["destination"])
                sc["lat2"], sc["lon2"] = ap1["lat"], ap1["lon"]
            else:
                sc["lat2"], sc["lon2"] = sc["destination"]
            if self.lat0 is None:
                self.lat0 = (sc["lat1"] + sc["lat2"]) / 2
                self.lon0 = (sc["lon1"] + sc["lon2"]) / 2
            else:
                self.lat0 = ((sc["lat1"] + sc["lat2"]) / 2 + self.lat0) / 2
                self.lon0 = ((sc["lon1"] + sc["lon2"]) / 2 + self.lon0) / 2
            sc["aircraft"] = oc.prop.aircraft(actype, use_synonym=use_synonym)
            sc["engtype"] = sc["aircraft"]["engine"]["default"]
            sc["engine"] = oc.prop.engine(sc["engtype"])
            sc["mass_init"] = sc.get("m0", 0.8) * sc["aircraft"]["mtow"]
            sc["oew"] = sc["aircraft"]["oew"]
            sc["mlw"] = sc["aircraft"]["mlw"]
            sc["fuel_max"] = sc["aircraft"]["mfc"]
            sc["mach_max"] = sc["aircraft"]["mmo"]

            sc["thrust"] = oc.Thrust(actype, use_synonym=use_synonym)
            sc["wrap"] = openap.WRAP(actype, use_synonym=use_synonym)
            sc["drag"] = oc.Drag(actype, wave_drag=True, use_synonym=use_synonym)
            sc["fuelflow"] = oc.FuelFlow(
                actype, wave_drag=True, use_synonym=use_synonym
            )
            sc["emission"] = oc.Emission(actype, use_synonym=use_synonym)

        # from pyproj import Proj
        # self.proj = Proj(
        #     proj="lcc",
        #     ellps="WGS84",
        #     lat_1=min(self.lat1, self.lat2),
        #     lat_2=max(self.lat1, self.lat2),
        #     lat_0=(self.lat1 + self.lat2) / 2,
        #     lon_0=(self.lon1 + self.lon2) / 2,
        # )

        # Check cruise range
        for sc in self.scenarios:
            sc["wind"] = None
            sc["range"] = oc.aero.distance(
                sc["lat1"], sc["lon1"], sc["lat2"], sc["lon2"]
            )

            max_range = sc["wrap"].cruise_range()["maximum"] * 1.2
            if sc["range"] > max_range * 1000:
                warnings.warn(f"{sc['actype']} destination likely out of cruise range")

            sc["debug"] = False

            self.setup(sc)

    def proj(self, lon, lat, inverse=False, symbolic=False):
        lat0 = self.lat0
        lon0 = self.lon0

        if not inverse:
            if symbolic:
                bearings = oc.aero.bearing(lat0, lon0, lat, lon) / 180 * 3.14159
                distances = oc.aero.distance(lat0, lon0, lat, lon)
                x = distances * ca.sin(bearings)
                y = distances * ca.cos(bearings)
            else:
                bearings = openap.aero.bearing(lat0, lon0, lat, lon) / 180 * 3.14159
                distances = openap.aero.distance(lat0, lon0, lat, lon)
                x = distances * np.sin(bearings)
                y = distances * np.cos(bearings)

            return x, y
        else:
            x, y = lon, lat
            if symbolic:
                distances = ca.sqrt(x**2 + y**2)
                bearing = ca.arctan2(x, y) * 180 / 3.14159
                lat, lon = oc.aero.latlon(lat0, lon0, distances, bearing)
            else:
                distances = np.sqrt(x**2 + y**2)
                bearing = np.arctan2(x, y) * 180 / 3.14159
                lat, lon = openap.aero.latlon(lat0, lon0, distances, bearing)

            return lon, lat

    def initial_guess(self, sc, flight: pd.DataFrame = None):
        m_guess = sc["mass_init"] * np.ones(sc["nodes"] + 1)
        ts_guess = np.linspace(0, 6 * 3600, sc["nodes"] + 1)

        if flight is None:
            h_cr = sc["aircraft"]["cruise"]["height"]
            xp_0, yp_0 = self.proj(sc["lon1"], sc["lat1"])
            xp_f, yp_f = self.proj(sc["lon2"], sc["lat2"])
            xp_guess = np.linspace(xp_0, xp_f, sc["nodes"] + 1)
            yp_guess = np.linspace(yp_0, yp_f, sc["nodes"] + 1)
            h_guess = h_cr * np.ones(sc["nodes"] + 1)
        else:
            xp_guess, yp_guess = self.proj(flight.longitude, flight.latitude)
            h_guess = flight.altitude * ft
            if "mass" in flight:
                m_guess = flight.mass

            if "ts" in flight:
                ts_guess = flight.ts
            elif "timestamp" in flight:
                ts_guess = (
                    flight.timestamp - flight.timestamp.min()
                ).dt.total_seconds()

        return np.vstack([xp_guess, yp_guess, h_guess, m_guess, ts_guess]).T

    def enable_wind(self, sc, windfield: pd.DataFrame):
        sc["wind"] = tools.PolyWind(
            windfield,
            lambda lon, lat, **kw: self.proj(lon, lat, **kw),
            sc["lat1"],
            sc["lon1"],
            sc["lat2"],
            sc["lon2"],
        )

    def change_engine(self, sc, engtype):
        sc["engtype"] = engtype
        sc["engine"] = oc.prop.engine(engtype)

        sc["thrust"] = oc.Thrust(
            sc["actype"],
            engtype,
            use_synonym=self.use_synonym,
            force_engine=True,
        )

        sc["fuelflow"] = oc.FuelFlow(
            sc["actype"],
            engtype,
            wave_drag=True,
            use_synonym=self.use_synonym,
            force_engine=True,
        )

        sc["emission"] = oc.Emission(
            sc["actype"], engtype, use_synonym=self.use_synonym
        )

    def collocation_coeff(self):
        # Get collocation points using Legendre polynomials
        tau_root = np.append(0, ca.collocation_points(self.polydeg, "legendre"))

        # C[i,j] = time derivative of Lagrange polynomial i evaluated at collocation point j
        C = np.zeros((self.polydeg + 1, self.polydeg + 1))

        # D[j] = Lagrange polynomial j evaluated at final time (t=1)
        D = np.zeros(self.polydeg + 1)

        # B[j] = integral of Lagrange polynomial j from 0 to 1
        B = np.zeros(self.polydeg + 1)

        # For each collocation point, construct Lagrange polynomial and calculate coefficients
        for j in range(self.polydeg + 1):
            # Construct Lagrange polynomial that is 1 at tau_root[j] and 0 at tau_root[r] where r != j
            p = np.poly1d([1])
            for r in range(self.polydeg + 1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

            # Evaluate polynomial at t=1 for continuity constraints
            D[j] = p(1.0)

            # Get time derivative coefficients for collocation constraints
            pder = np.polyder(p)
            for r in range(self.polydeg + 1):
                C[j, r] = pder(tau_root[r])

            # Get integral coefficients for cost function quadrature
            pint = np.polyint(p)
            B[j] = pint(1.0)

        return C, D, B

    def xdot(self, sc, x, u) -> ca.MX:
        """Ordinary differential equation for cruising

        Args:
            x (ca.MX): States [x position (m), y position (m), height (m), mass (kg)]
            u (ca.MX): Controls [mach number, vertical speed (m/s), heading (rad)]

        Returns:
            ca.MX: State direvatives
        """
        xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]
        mach, vs, psi = u[0], u[1], u[2]

        v = oc.aero.mach2tas(mach, h, dT=self.dT)
        gamma = ca.arctan2(vs, v)

        dx = v * ca.sin(psi) * ca.cos(gamma)
        dy = v * ca.cos(psi) * ca.cos(gamma)

        if sc.get("wind") is not None:
            dx += sc["wind"].calc_u(xp, yp, h, ts)
            dy += sc["wind"].calc_v(xp, yp, h, ts)

        dh = vs

        dm = -sc["fuelflow"].enroute(m, v / kts, h / ft, vs / fpm, dT=self.dT)

        dt = 1

        return ca.vertcat(dx, dy, dh, dm, dt)

    def setup(
        self,
        sc,
        nodes: int | None = None,
        polydeg: int = 3,
        debug=False,
        ipopt_kwargs={},
        **kwargs,
    ):
        if nodes is not None:
            sc["nodes"] = nodes
        else:
            sc["nodes"] = int(sc["range"] / 50_000)  # node every 50km

        max_nodes = kwargs.get("max_nodes", 120)

        sc["nodes"] = max(20, sc["nodes"])
        # sc["nodes"] = min(max_nodes, sc["nodes"])

        self.polydeg = polydeg

        max_iteration = kwargs.get("max_iteration", kwargs.get("max_iterations", 3000))
        tol = kwargs.get("tol", 1e-6)
        acceptable_tol = kwargs.get("acceptable_tol", 1e-4)
        alpha_for_y = kwargs.get("alpha_for_y", "primal-and-full")
        hessian_approximation = kwargs.get("hessian_approximation", "limited-memory")

        self.debug = debug

        if debug:
            print("Calculating optimal trajectory...")
            ipopt_print = 5
            print_time = 1
        else:
            ipopt_print = 0
            print_time = 0

        self.solver_options = {
            "print_time": print_time,
            "calc_lam_p": False,
            "ipopt.print_level": ipopt_print,
            "ipopt.sb": "yes",
            "ipopt.max_iter": max_iteration,
            "ipopt.fixed_variable_treatment": "relax_bounds",
            "ipopt.tol": tol,
            "ipopt.acceptable_tol": acceptable_tol,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.alpha_for_y": alpha_for_y,
            "ipopt.hessian_approximation": hessian_approximation,
        }

        for key, value in ipopt_kwargs.items():
            self.solver_options[f"ipopt.{key}"] = value

    def init_model(self, sc, objective, **kwargs):
        autoscale_cost = kwargs.get("auto_scale_cost", False)
        # Model variables
        xp = ca.MX.sym("xp")
        yp = ca.MX.sym("yp")
        h = ca.MX.sym("h")
        m = ca.MX.sym("m")
        ts = ca.MX.sym("ts")

        mach = ca.MX.sym("mach")
        vs = ca.MX.sym("vs")
        psi = ca.MX.sym("psi")

        sc["x"] = ca.vertcat(xp, yp, h, m, ts)
        sc["u"] = ca.vertcat(mach, vs, psi)

        sc["ts_final"] = ca.MX.sym("ts_final")

        # Control discretization
        sc["dt"] = sc["ts_final"] / sc["nodes"]

        # Handel objective function
        if isinstance(objective, Callable):
            sc["objective"] = objective
        elif objective.lower().startswith("ci:"):
            ci = int(objective[3:])
            kwargs["ci"] = ci
            sc["objective"] = self.obj_ci
        else:
            sc["objective"] = getattr(self, f"obj_{objective}")

        L = sc["objective"](sc, sc["x"], sc["u"], sc["dt"], **kwargs)

        if autoscale_cost:
            # scale objective based on initial guess
            x0 = sc["x_guess"].T
            u0 = sc["u_guess"]
            dt0 = sc["range"] / 200 / sc["nodes"]
            cost = np.sum(sc["objective"](x0, u0, dt0, symbolic=False, **kwargs))
            L = L / cost * 1e3

        # Continuous time dynamics
        sc["func_dynamics"] = ca.Function(
            f"f_sc_{sc['id']}",
            [sc["x"], sc["u"]],
            [self.xdot(sc, sc["x"], sc["u"]), L],
            ["x", "u"],
            ["xdot", "L"],
            {"allow_free": True},
        )

    def interpolate_state(self, X, k, Xc, dt, tau):
        """
        Interpolate state inside collocation interval using Lagrange polynomials.
        X   : state at interval start
        Xc  : list of collocation states
        dt  : interval duration
        tau : normalized time in [0,1]
        """
        # Lagrange basis (polydeg = 3 example)
        L = [None] * (self.polydeg + 1)
        tau_root = np.append(0, ca.collocation_points(self.polydeg, "legendre"))

        for j in range(self.polydeg + 1):
            lj = 1
            for r in range(self.polydeg + 1):
                if r != j:
                    lj *= (tau - tau_root[r]) / (tau_root[j] - tau_root[r])
            L[j] = lj

        x_tau = L[0] * X
        for j in range(self.polydeg):
            x_tau += L[j + 1] * Xc[j]

        return x_tau

    def interpolate_state_global(self, X, Xc, dt, t0, t, nodes):
        """
        Interpolates trajectory at absolute time t
        """
        x_interp = 0

        for k in range(nodes):

            t_k0 = t0 + k * dt
            t_k1 = t0 + (k + 1) * dt

            # smooth activation (≈ indicator)
            wk = 0.5 * (ca.tanh(50 * (t - t_k0)) - ca.tanh(50 * (t - t_k1)))

            tau = (t - t_k0) / dt

            xk = self.interpolate_state(
                X[k],
                k,
                Xc[k],
                dt,
                tau,
            )

            x_interp += wk * xk

        return x_interp

    def _calc_emission(self, x, u, symbolic=True):
        xp, yp, h, m = x[0], x[1], x[2], x[3]
        mach, vs, psi = u[0], u[1], u[2]

        if symbolic:
            fuelflow = self.fuelflow
            emission = self.emission
            v = oc.aero.mach2tas(mach, h, dT=self.dT)
        else:
            fuelflow = openap.FuelFlow(
                self.actype, self.engtype, polydeg=2, use_synonym=self.use_synonym
            )
            emission = openap.Emission(
                self.actype, self.engtype, use_synonym=self.use_synonym
            )
            v = openap.aero.mach2tas(mach, h, dT=self.dT)

        ff = fuelflow.enroute(m, v / kts, h / ft, vs / fpm, dT=self.dT)
        co2 = emission.co2(ff)
        h2o = emission.h2o(ff)
        sox = emission.sox(ff)
        soot = emission.soot(ff)
        nox = emission.nox(ff, v / kts, h / ft, dT=self.dT)

        return co2, h2o, sox, soot, nox

    def obj_fuel(self, sc, x, u, dt, symbolic=True, **kwargs):
        xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]
        mach, vs, psi = u[0], u[1], u[2]

        if symbolic:
            fuelflow = sc["fuelflow"]
            v = oc.aero.mach2tas(mach, h, dT=self.dT)
        else:
            fuelflow = openap.FuelFlow(
                sc["actype"],
                sc["engtype"],
                use_synonym=self.use_synonym,
                force_engine=True,
            )
            v = openap.aero.mach2tas(mach, h, dT=self.dT)

        ff = fuelflow.enroute(m, v / kts, h / ft, vs / fpm, dT=self.dT)
        return ff * dt

    def obj_time(self, x, u, dt, **kwargs):
        return dt

    def obj_ci(self, x, u, dt, ci, time_price=25, fuel_price=0.8, **kwargs):
        """
        Calculate the objective cost index (CI) based on time and fuel costs.

        Parameters:
        x (ca.MX): state vector.
        u (ca.MX): control vector.
        dt (ca.MX): time step.
        ci (float): Cost index, a percentage value between 0 and 100.
        time_price (float): optional, cost of time per minute (default is 25 EUR/min).
        fuel_price (float): optional, cost of fuel per liter (default is 0.8 EUR/L).

        Returns:
        ca.MX: cost index objective.
        """

        fuel = self.obj_fuel(x, u, dt, **kwargs)

        # time cost 25 eur/min
        time_cost = (dt / 60) * time_price

        # fuel cost 0.8 eur/L, Jet A density 0.82
        fuel_cost = fuel * (fuel_price / 0.82)

        obj = ci / 100 * time_cost + (1 - ci / 100) * fuel_cost
        return obj

    def obj_gwp20(self, x, u, dt, **kwargs):
        co2, h2o, sox, soot, nox = self._calc_emission(x, u, **kwargs)
        cost = co2 + 0.22 * h2o + 619 * nox - 832 * sox + 4288 * soot
        # cost = cost * 1e-3
        return cost * dt

    def obj_gwp50(self, x, u, dt, **kwargs):
        co2, h2o, sox, soot, nox = self._calc_emission(x, u, **kwargs)
        cost = co2 + 0.1 * h2o + 205 * nox - 392 * sox + 2018 * soot
        # cost = cost * 1e-3
        return cost * dt

    def obj_gwp100(self, x, u, dt, **kwargs):
        co2, h2o, sox, soot, nox = self._calc_emission(x, u, **kwargs)
        cost = co2 + 0.06 * h2o + 114 * nox - 226 * sox + 1166 * soot
        # cost = cost * 1e-3
        return cost * dt

    def obj_gtp20(self, x, u, dt, **kwargs):
        co2, h2o, sox, soot, nox = self._calc_emission(x, u, **kwargs)
        cost = co2 + 0.07 * h2o - 222 * nox - 241 * sox + 1245 * soot
        # cost = cost * 1e-3
        return cost * dt

    def obj_gtp50(self, x, u, dt, **kwargs):
        co2, h2o, sox, soot, nox = self._calc_emission(x, u, **kwargs)
        cost = co2 + 0.01 * h2o - 69 * nox - 38 * sox + 195 * soot
        # cost = cost * 1e-3
        return cost * dt

    def obj_gtp100(self, x, u, dt, **kwargs):
        co2, h2o, sox, soot, nox = self._calc_emission(x, u, **kwargs)
        cost = co2 + 0.008 * h2o + 13 * nox - 31 * sox + 161 * soot
        # cost = cost * 1e-3
        return cost * dt

    def obj_grid_cost(self, x, u, dt, **kwargs):
        """
        Calculate the cost of the grid object.

        Parameters:
        x (ca.MX): State vector [xp, yp, h, m, ts].
        u (ca.MX): Control vector [mach, vs, psi].
        dt (ca.MX): Time step.

        **kwargs (dict): Additional keyword arguments.
            - interpolant (function): Interpolant function.
            - symbolic (bool): Flag indicating whether to use symbolic computation.
            - n_dim (int): Dimension of the input data (3 or 4), default to 3.
            - time_dependent (bool): Flag indicating whether the cost is time dependent.
            The cost will be multiplied by dt if true.

        Returns:
        cost (ca.MX): cost objective.

        Raises:
        AssertionError: If n_dim is not 3 or 4.
        """

        xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]

        interpolant = kwargs.get("interpolant", None)
        symbolic = kwargs.get("symbolic", True)
        n_dim = kwargs.get("n_dim", 3)
        time_dependent = kwargs.get("time_dependent", True)
        assert n_dim in [3, 4]

        self.solver_options["ipopt.hessian_approximation"] = "limited-memory"

        lon, lat = self.proj(xp, yp, inverse=True, symbolic=symbolic)

        if n_dim == 3:
            input_data = [lon, lat, h]
        else:
            input_data = [lon, lat, h, ts]

        if symbolic:
            input_data = ca.vertcat(*input_data)
        else:
            input_data = np.array(input_data)

        cost = interpolant(input_data)

        if not symbolic:
            cost = cost.full()[0]

        if time_dependent:
            cost *= dt

        return cost

    def obj_combo(self, x, u, dt, obj1, obj2, ratio=0.5, **kwargs):
        if isinstance(obj1, str):
            obj1 = getattr(self, f"obj_{obj1}")

        if isinstance(obj2, str):
            obj2 = getattr(self, f"obj_{obj2}")

        x0 = self.x_guess.T
        u0 = self.u_guess
        dt0 = self.range / 200 / sc["nodes"]

        kwargs_ = kwargs.copy()
        kwargs_["symbolic"] = False

        n1 = obj1(x0, u0, dt0, **kwargs_).sum()
        n2 = obj2(x0, u0, dt0, **kwargs_).sum()

        c1 = obj1(x, u, dt, **kwargs)
        c2 = obj2(x, u, dt, **kwargs)

        return ratio * c1 / n1 + (1 - ratio) * c2 / n2

    def to_trajectory(self, sc, ts_final, x_opt, u_opt):
        X = x_opt.full()
        U = u_opt.full()

        # Extrapolate the final control point, Uf
        U2 = U[:, -2:-1]
        U1 = U[:, -1:]
        Uf = U1 + (U1 - U2)

        U = np.append(U, Uf, axis=1)
        n = sc["nodes"] + 1

        sc["X"] = X
        sc["U"] = U
        sc["dt"] = ts_final / (n - 1)

        xp, yp, h, mass, ts = X
        mach, vs, psi = U
        lon, lat = self.proj(xp, yp, inverse=True)
        ts_ = np.linspace(0, ts_final, n).round(4) + sc["tstart"]
        tas = (openap.aero.mach2tas(mach, h, dT=self.dT) / kts).round(4)
        alt = (h / ft).round()
        vertrate = (vs / fpm).round()

        df = pd.DataFrame(
            dict(
                mass=mass,
                ts=ts_,
                x=xp,
                y=yp,
                h=h,
                latitude=lat,
                longitude=lon,
                altitude=alt,
                mach=mach.round(6),
                tas=tas,
                vertical_rate=vertrate,
                heading=(np.rad2deg(psi) % 360).round(4),
            )
        )

        fuelflow = openap.FuelFlow(
            sc["actype"],
            sc["engtype"],
            use_synonym=self.use_synonym,
            force_engine=True,
        )

        df = df.assign(
            fuelflow=(
                fuelflow.enroute(
                    mass=df.mass, tas=tas, alt=alt, vs=vertrate, dT=self.dT
                )
            )
        )
        return df
        # if self.wind:
        #     wu = self.wind.calc_u(xp, yp, h, ts)
        #     wv = self.wind.calc_v(xp, yp, h, ts)
        #     df = df.assign(wu=wu, wv=wv)
