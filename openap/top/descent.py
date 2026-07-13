from math import pi
import openap
import casadi as ca
import numpy as np
import openap.casadi as oc
import pandas as pd
from openap.extra.aero import fpm, ft, kts

from .base import Base
from .cruise import Cruise


class Descent(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curise = Cruise(*args, **kwargs)

    def init_conditions(self, sc, df_cruise, descent_guess=None, **kwargs):
        """Initialize direct collocation bounds and guesses."""

        alt_min = kwargs.get("alt_min", None)
        alt_start = kwargs.get("alt_start", None)
        if alt_min is None:
            h_min = 100 * ft
        else:
            h_min = alt_min * ft
        od_bearing = openap.aero.bearing(sc["lat1"], sc["lon1"], sc["lat2"], sc["lon2"])
        od_psi = od_bearing * pi / 180

        xp_0, yp_0 = self.proj(sc["lon1"], sc["lat1"])
        xp_f, yp_f = self.proj(sc["lon2"], sc["lat2"])

        x_min = min(xp_0, xp_f) - 30_000
        x_max = max(xp_0, xp_f) + 30_000
        y_min = min(yp_0, yp_f) - 30_000
        y_max = max(yp_0, yp_f) + 30_000
        ts_min = 0
        ts_max = 6 * 3600

        # df_cruise = kwargs.get("df_cruise", None)
        mass_oew = sc["oew"]
        mass_tod = sc["mass_init"]
        if df_cruise is None:
            cruise_mach = sc["aircraft"]["mmo"]
        else:
            cruise_mach = df_cruise.mach.max()

        if alt_start is not None:
            h_start = alt_start * ft
            if df_cruise is not None and h_start > df_cruise.h.iloc[0]:
                print(
                    "The given alt_start is beyond performance limit, "
                    f"we will use {df_cruise.h.iloc[0] / ft}"
                )
                h_start = df_cruise.h.iloc[0]
        else:
            if df_cruise is not None:
                h_start = df_cruise.h.iloc[-1]
            else:
                h_start = sc["aircraft"]["ceiling"]

        trk_start_0 = kwargs.get("trk_start", None)
        if trk_start_0 is None:
            trk_start = od_bearing
        else:
            trk_start = trk_start_0
        psi_start = trk_start * pi / 180

        runway_dir = kwargs.get("runway_dir", None)
        if runway_dir is None:
            psi_end = od_psi
        elif runway_dir <= 180 and trk_start >= 180 and (trk_start - runway_dir) > 180:
            psi_end = runway_dir * pi / 180
            psi_end = 2 * pi + psi_end
        elif runway_dir >= 180 and trk_start <= 180 and (runway_dir - trk_start) > 180:
            psi_end = runway_dir * pi / 180
            psi_start = 2 * pi + psi_start
        else:
            psi_end = runway_dir * pi / 180

        # Initial conditions - Lower and upper bounds
        sc["x_0_lb"] = [xp_0, yp_0, h_start - 100, mass_tod, ts_min]
        sc["x_0_ub"] = [xp_0, yp_0, h_start + 100, mass_tod, ts_min]

        # Final conditions - Lower and upper bounds
        sc["x_f_lb"] = [xp_f, yp_f, h_min, mass_oew, ts_min]
        sc["x_f_ub"] = [xp_f, yp_f, h_min, mass_tod, ts_max]

        # States - Lower and upper bounds
        sc["x_lb"] = [x_min, y_min, h_min, mass_oew, ts_min]
        sc["x_ub"] = [x_max, y_max, h_start + 100, mass_tod, ts_max]

        # States - guesses
        n1 = sc["nodes"] + 1
        if descent_guess is not None:
            g = descent_guess
            s_src = np.linspace(0, 1, len(g))
            s_dst = np.linspace(0, 1, n1)
            xpg, ypg = self.proj(g.longitude.values, g.latitude.values)
            xp_guess = np.interp(s_dst, s_src, xpg)
            yp_guess = np.interp(s_dst, s_src, ypg)
            h_guess = np.interp(s_dst, s_src, g.altitude.values * ft)
            m_guess = np.interp(s_dst, s_src, g.mass.values)
            ts_guess = np.interp(s_dst, s_src, g.ts.values - g.ts.values[0])
        else:
            xp_guess = np.linspace(xp_0, xp_f, n1)
            yp_guess = np.linspace(yp_0, yp_f, n1)
            h_guess = np.linspace(h_start, h_min, n1)
            m_guess = mass_tod * np.ones(n1)
            ts_guess = np.linspace(0, 6 * 3600, n1)
        sc["x_guess"] = np.vstack([xp_guess, yp_guess, h_guess, m_guess, ts_guess]).T

        # Control init - lower and upper bounds
        sc["u_0_lb"] = [cruise_mach - 0.2, -3000 * fpm, psi_start - pi / 4]
        sc["u_0_ub"] = [cruise_mach + 0.2, 500 * fpm, psi_start + pi / 4]

        # Control final - lower and upper bounds
        if runway_dir is None:
            sc["u_f_lb"] = [0.1, -3500 * fpm, psi_end - pi / 2]
            sc["u_f_ub"] = [0.3, 0 * fpm, psi_end + pi / 2]
        else:
            sc["u_f_lb"] = [0.1, -3500 * fpm, psi_end]
            sc["u_f_ub"] = [0.3, 0 * fpm, psi_end]

        # Control - Lower and upper bound
        sc["u_lb"] = [0.1, -3500 * fpm, np.min([psi_end, psi_start]) - pi / 2]
        sc["u_ub"] = [
            cruise_mach + 0.2,
            1000 * fpm,
            np.max([psi_end, psi_start]) + pi / 2,
        ]

        # Control - guesses
        sc["u_0_guess"] = [0.7, 0 * fpm, psi_start]
        sc["u_guess"] = [0.4, -2000 * fpm, od_psi]
        sc["u_f_guess"] = [0.3, -1000 * fpm, psi_end]

    def trajectory(self, objective="fuel", df_cruise=None, **kwargs) -> pd.DataFrame:
        conflict_guess = kwargs.get("conflict_guess", None)

        if df_cruise is None:
            if self.debug:
                print("Finding the preliminary optimal cruise trajectory parameters...")
            df_cruise = self.curise.trajectory(objective, **kwargs)

        if self.debug:
            print("Calculating optimal descent trajectory...")

        for isc, sc in enumerate(self.scenarios):
            guess_isc = conflict_guess[isc] if conflict_guess is not None else None
            self.init_conditions(sc, df_cruise[isc], descent_guess=guess_isc, **kwargs)
            self.init_model(sc, objective, **kwargs)

        runway_dir = kwargs.get("runway_dir", None)

        C, D, B = self.collocation_coeff()

        # Start with an empty NLP
        w = []  # Containing all the states & controls generated
        w0 = []  # Containing the initial guess for w
        lbw = []  # Lower bound constraints on the w variable
        ubw = []  # Upper bound constraints on the w variable
        J = 0  # Objective function
        g = []  # Constraint function
        lbg = []  # Constraint lb value
        ubg = []  # Constraint ub value

        nsc = len(self.scenarios)
        # For plotting x and u given w
        X = [[] for _ in range(nsc)]
        U = [[] for _ in range(nsc)]
        Xc_store = [[] for _ in range(nsc)]

        # Apply initial conditions
        # Create Xk such that it is the same length as x
        for isc, sc in enumerate(self.scenarios):
            nodes = sc["nodes"]
            nstates = sc["x"].shape[0]
            Xk = ca.MX.sym(f"X_{isc}_0", nstates, sc["x"].shape[1])
            w.append(Xk)
            lbw.append(sc["x_0_lb"])
            ubw.append(sc["x_0_ub"])
            w0.append(sc["x_guess"][0])
            X[isc].append(Xk)

            # Formulate the NLP
            for k in range(nodes):
                # New NLP variable for the control
                Uk = ca.MX.sym(f"U_{isc}_{k}", sc["u"].shape[0])
                U[isc].append(Uk)
                w.append(Uk)
                if k == 0:
                    lbw.append(sc["u_0_lb"])
                    ubw.append(sc["u_0_ub"])
                    w0.append(sc["u_0_guess"])
                elif k == nodes - 1:
                    lbw.append(sc["u_f_lb"])
                    ubw.append(sc["u_f_ub"])
                    w0.append(sc["u_f_guess"])
                else:
                    lbw.append(sc["u_lb"])
                    ubw.append(sc["u_ub"])
                    w0.append(sc["u_guess"])

                # State at collocation points
                Xc = []
                for j in range(self.polydeg):
                    Xkj = ca.MX.sym(f"X_{isc}_{k}_{j}", nstates)
                    Xc.append(Xkj)
                    w.append(Xkj)
                    lbw.append(sc["x_lb"])
                    ubw.append(sc["x_ub"])
                    w0.append(sc["x_guess"][k])

                # Loop over collocation points
                Xk_end = D[0] * Xk
                for j in range(1, self.polydeg + 1):
                    # Expression for the state derivative at the collocation point
                    xpc = C[0, j] * Xk
                    for r in range(self.polydeg):
                        xpc = xpc + C[r + 1, j] * Xc[r]

                    # Append collocation equations
                    fj, qj = sc["func_dynamics"](Xc[j - 1], Uk)
                    g.append(sc["dt"] * fj - xpc)
                    lbg.append([0] * nstates)
                    ubg.append([0] * nstates)

                    # Add contribution to the end state
                    Xk_end = Xk_end + D[j] * Xc[j - 1]

                    # Add contribution to quadrature function
                    # J = J + B[j] * qj * dt
                    J = J + B[j] * qj

                # New NLP variable for state at end of interval
                Xk = ca.MX.sym(f"X_{isc}_{k+1}", nstates)
                w.append(Xk)
                X[isc].append(Xk)
                Xc_store[isc].append(Xc)

                if k < nodes - 1:
                    # normal boundary conditions
                    lbw.append(sc["x_lb"])
                    ubw.append(sc["x_ub"])
                else:
                    # Final bounday conditions
                    lbw.append(sc["x_f_lb"])
                    ubw.append(sc["x_f_ub"])

                w0.append(sc["x_guess"][k])

                # Add equality constraint
                g.append(Xk_end - Xk)
                lbg.append([0] * nstates)
                ubg.append([0] * nstates)

            w.append(sc["ts_final"])
            lbw.append([0])
            ubw.append([ca.inf])
            w0.append([3600])

            # constrain time and dt
            for k in range(1, nodes):
                g.append(X[isc][k][4] - X[isc][k - 1][4] - sc["dt"])
                lbg.append([0])
                ubg.append([0])

            # cas constraint
            for k in range(nodes):
                cas = oc.aero.mach2cas(U[isc][k][0], X[isc][k][2], dT=self.dT)
                cas_max = sc["aircraft"]["vmo"] * kts
                g.append(cas)
                lbg.append([0])
                ubg.append([cas_max])

            # smooth cas change
            for k in range(1, nodes):
                cask = oc.aero.mach2cas(U[isc][k][0], X[isc][k][2], dT=self.dT)
                cask1 = oc.aero.mach2cas(U[isc][k - 1][0], X[isc][k - 1][2], dT=self.dT)
                g.append((cask - cask1) / sc["dt"])
                lbg.append([-1])  # per second
                ubg.append([1])  # per second

            # smooth vertical rate changes
            for k in range(1, nodes):
                g.append((U[isc][k][1] - U[isc][k - 1][1]) / sc["dt"])
                lbg.append([-12 * fpm])
                ubg.append([12 * fpm])  # to be tunned

            # smooth heading changes
            for k in range(1, nodes):
                g.append((U[isc][k][2] - U[isc][k - 1][2]) / sc["dt"])
                lbg.append([-1.5 * pi / 180])
                ubg.append([1.5 * pi / 180])

            # last nodes along the runway_dir
            if runway_dir is not None:
                for k in range(nodes - 3, nodes):
                    g.append((U[isc][k][2] - U[isc][k - 1][2]))
                    lbg.append([-0.5 * pi / 180])
                    ubg.append([0.5 * pi / 180])

            # force and energy constraint
            for k in range(nodes):
                S = sc["aircraft"]["wing"]["area"]
                cd0 = sc["drag"].polar["clean"]["cd0"]
                ck = sc["drag"].polar["clean"]["k"]
                mass = X[isc][k][3]
                v = oc.aero.mach2tas(U[isc][k][0], X[isc][k][2], dT=self.dT)
                tas = v / kts
                alt = X[isc][k][2] / ft
                rho = oc.aero.density(X[isc][k][2], dT=self.dT)
                thrust_max = sc["thrust"].cruise(tas, alt, dT=self.dT)
                drag = sc["drag"].clean(mass, tas, alt, dT=self.dT)

                # max_thrust > drag (5% margin)
                g.append(thrust_max * 0.95 - drag)
                lbg.append([0])
                ubg.append([ca.inf])

                # max lift * 80% > weight (20% margin)
                drag_max = thrust_max * 0.9
                cd_max = drag_max / (0.5 * rho * v**2 * S + 1e-10)
                cd0 = sc["drag"].polar["clean"]["cd0"]
                ck = sc["drag"].polar["clean"]["k"]
                cl_max = ca.sqrt(ca.fmax(1e-10, (cd_max - cd0) / ck))
                L_max = cl_max * 0.5 * rho * v**2 * S
                g.append(L_max * 0.8 - mass * oc.aero.g0)
                lbg.append([0])
                ubg.append([ca.inf])

                # excess energy > change potential energy
                excess_energy = (thrust_max - drag) * v - mass * oc.aero.g0 * U[isc][k][
                    1
                ]
                g.append(excess_energy)
                lbg.append([0])
                ubg.append([ca.inf])

        if len(self.scenarios) > 1 and self.conflict:
            Rxy = 5.0 * oc.aero.nm
            Rz = 1000 * oc.aero.ft

            guess = conflict_guess
            if guess is not None:
                pair_times = self.candidate_sync_times(guess)
            else:
                # fallback: coarse overlap grid per pair (no warm start to detect from)
                pair_times = {}
                for i in range(len(self.scenarios)):
                    for j in range(i + 1, len(self.scenarios)):
                        sci, scj = self.scenarios[i], self.scenarios[j]
                        t0 = max(sci["tstart"], scj["tstart"])
                        t1 = min(
                            sci["tstart"] + sci["range"] / 200,
                            scj["tstart"] + scj["range"] / 200,
                        )
                        if t1 > t0:
                            pair_times[(i, j)] = np.linspace(t0, t1, 60)

            for (i, j), tsync in pair_times.items():
                sci, scj = self.scenarios[i], self.scenarios[j]
                for t in tsync:
                    xi = self.interpolate_state_global(
                        X[i], Xc_store[i], sci["dt"], sci["tstart"], t, sci["nodes"]
                    )
                    xj = self.interpolate_state_global(
                        X[j], Xc_store[j], scj["dt"], scj["tstart"], t, scj["nodes"]
                    )
                    dist_x = xi[0] - xj[0]
                    dist_y = xi[1] - xj[1]
                    dist_z = xi[2] - xj[2]
                    ellipsoid = (
                        (dist_x / Rxy) ** 2 + (dist_y / Rxy) ** 2 + (dist_z / Rz) ** 8
                    )
                    g.append(ellipsoid)
                    lbg.append([1.3])
                    ubg.append([ca.inf])
        # Concatenate vectors
        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        # X = ca.horzcat(*X)
        # U = ca.horzcat(*U)
        self.X_store = X
        self.U_store = U
        w0 = np.concatenate(w0)
        lbw = np.concatenate(lbw)
        ubw = np.concatenate(ubw)
        lbg = np.concatenate(lbg)
        ubg = np.concatenate(ubg)

        # Create an NLP solver
        nlp = {"f": J, "x": w, "g": g}

        self.solver = ca.nlpsol("solver", "ipopt", nlp, self.solver_options)

        self.solution = self.solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # final timestep
        w_opt = self.solution["x"]

        trajectories = []
        # Function to get x and u from w
        for isc, sc in enumerate(self.scenarios):
            Xi = ca.horzcat(*self.X_store[isc])  # (nstates, N+1)
            Ui = ca.horzcat(*self.U_store[isc])  # (ncontrols, N)
            output_i = ca.Function(
                f"output_sc_{isc}",
                [w],
                [Xi, Ui],
                [f"w_{isc}"],
                [f"x_{isc}", f"u_{isc}"],
            )

            x_opt_i, u_opt_i = output_i(w_opt)

            ts_final_i = x_opt_i.full()[4, -1]
            df_i = self.to_trajectory(sc, ts_final_i, x_opt_i, u_opt_i)
            df_i["scenario"] = isc
            trajectories.append(df_i)
        return trajectories
