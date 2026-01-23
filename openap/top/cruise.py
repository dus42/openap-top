import warnings
from math import pi

import casadi as ca
import numpy as np
import openap.casadi as oc
import pandas as pd
from openap.extra.aero import fpm, ft, kts

from .base import Base


class Cruise(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fix_mach = False
        self.fix_alt = False
        self.fix_track = False
        self.allow_descent = False

    def fix_mach_number(self):
        self.fix_mach = True

    def fix_cruise_altitude(self):
        self.fix_alt = True

    def fix_track_angle(self):
        self.fix_track = True

    def allow_cruise_descent(self):
        self.allow_descent = True

    def init_conditions(self, sc, **kwargs):
        """Initialize direct collocation bounds and guesses."""

        # Convert lat/lon to cartisian coordinates.
        xp_0, yp_0 = self.proj(sc["lon1"], sc["lat1"])
        xp_f, yp_f = self.proj(sc["lon2"], sc["lat2"])
        x_min = min(xp_0, xp_f) - 10_000
        x_max = max(xp_0, xp_f) + 10_000
        y_min = min(yp_0, yp_f) - 10_000
        y_max = max(yp_0, yp_f) + 10_000

        ts_min = 0
        ts_max = max(5, sc["range"] / 1000 / 500) * 3600

        h_max = kwargs.get("h_max", sc["aircraft"]["limits"]["ceiling"])
        h_min = kwargs.get("h_min", 15_000 * ft)

        hdg = oc.aero.bearing(sc["lat1"], sc["lon1"], sc["lat2"], sc["lon2"])
        psi = hdg * pi / 180

        # Initial conditions - Lower upper bounds
        sc["x_0_lb"] = [xp_0, yp_0, h_min, sc["mass_init"], ts_min]
        sc["x_0_ub"] = [xp_0, yp_0, h_max, sc["mass_init"], ts_min]

        # Final conditions - Lower and upper bounds
        sc["x_f_lb"] = [xp_f, yp_f, h_min, sc["oew"], ts_min]
        sc["x_f_ub"] = [xp_f, yp_f, h_max, sc["mass_init"], ts_max]

        # States - Lower and upper bounds
        sc["x_lb"] = [x_min, y_min, h_min, sc["oew"], ts_min]
        sc["x_ub"] = [x_max, y_max, h_max, sc["mass_init"], ts_max]

        # Control init - lower and upper bounds
        sc["u_0_lb"] = [0.5, -500 * fpm, psi - pi / 4]
        sc["u_0_ub"] = [sc["mach_max"], 500 * fpm, psi + pi / 4]

        # Control final - lower and upper bounds
        sc["u_f_lb"] = [0.5, -500 * fpm, psi - pi / 4]
        sc["u_f_ub"] = [sc["mach_max"], 500 * fpm, psi + pi / 4]

        # Control - Lower and upper bound
        sc["u_lb"] = [0.5, -500 * fpm, psi - pi / 2]
        sc["u_ub"] = [sc["mach_max"], 500 * fpm, psi + pi / 2]

        # Initial guess - states
        sc["x_guess"] = self.initial_guess(sc)

        # Initial guess - controls
        sc["u_guess"] = [0.7, 0, psi]
        return sc

    def trajectory(self, objective="fuel", **kwargs) -> pd.DataFrame:
        """
        Computes the optimal trajectory for the aircraft based on the given objective.

        Parameters:
        - objective (str): The objective of the optimization, default is "fuel".
        - **kwargs: Additional keyword arguments.
            - max_fuel (float): Customized maximum fuel constraint.
            - initial_guess (pd.DataFrame): Initial guess for the trajectory. This is
                usually a exsiting flight trajectory.
            - return_failed (bool): If True, returns the DataFrame even if the
                optimization fails. Default is False.

        Returns:
        - pd.DataFrame: A DataFrame containing the optimized trajectory.

        Note:
        - The function uses CasADi for symbolic computation and optimization.
        - The constraints and bounds are defined based on the aircraft's performance
            and operational limits.
        """

        # arguments passed init_condition to overwright h_min and h_max
        for isc, sc in enumerate(self.scenarios):
            self.init_conditions(sc, **kwargs)
            self.init_model(sc, objective, **kwargs)

            initial_guess = kwargs.get("initial_guess", None)
            if initial_guess is not None:
                sc["x_guess"] = self.initial_guess(initial_guess)

        customized_max_fuel = kwargs.get("max_fuel", None)
        return_failed = kwargs.get("return_failed", False)

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

        # For plotting x and u given w
        nsc = len(self.scenarios)

        X = [[] for _ in range(nsc)]
        U = [[] for _ in range(nsc)]
        Xc_store = [[] for _ in range(nsc)]
        for isc, sc in enumerate(self.scenarios):
            nodes = sc["nodes"]
            # Apply initial conditions
            # Create Xk such that it is the same length as x
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
                elif k == sc["nodes"] - 1:
                    lbw.append(sc["u_f_lb"])
                    ubw.append(sc["u_f_ub"])
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
                    J = J + (B[j] * qj)  # /sc["range"]

                # New NLP variable for state at end of interval
                Xk = ca.MX.sym(f"X_{isc}_{k+1}", nstates)
                w.append(Xk)
                X[isc].append(Xk)
                Xc_store[isc].append(Xc)

                # lbw.append(x_lb)
                # ubw.append(x_ub)

                if k < nodes - 1:
                    lbw.append(sc["x_lb"])
                    ubw.append(sc["x_ub"])
                else:
                    # Final conditions
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
            w0.append([sc["range"] * 1000 / 200])

            # aircraft performane constraints
            for k in range(nodes):
                S = sc["aircraft"]["wing"]["area"]
                mass = X[isc][k][3]
                v = oc.aero.mach2tas(U[isc][k][0], X[isc][k][2], dT=self.dT)
                tas = v / kts
                alt = X[isc][k][2] / ft
                rho = oc.aero.density(X[isc][k][2], dT=self.dT)
                thrust_max = sc["thrust"].cruise(tas, alt, dT=self.dT)

                # max_thrust * 95% > drag (5% margin)
                g.append(
                    thrust_max * 0.95 - sc["drag"].clean(mass, tas, alt, dT=self.dT)
                )
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

            # ts and dt should be consistent
            for k in range(nodes - 1):
                g.append(X[isc][k + 1][4] - X[isc][k][4] - sc["dt"])
                lbg.append([-1e-3])
                ubg.append([1e-3])

            # # smooth Mach number change
            # for k in range(self.nodes - 1):
            #     g.append(U[k + 1][0] - U[k][0])
            #     lbg.append([-0.2])
            #     ubg.append([0.2])  # to be tunned

            # # smooth vertical rate change
            # for k in range(self.nodes - 1):
            #     g.append(U[k + 1][1] - U[k][1])
            #     lbg.append([-500 * fpm])
            #     ubg.append([500 * fpm])  # to be tunned

            # smooth heading change
            for k in range(nodes - 1):
                g.append(U[isc][k + 1][2] - U[isc][k][2])
                lbg.append([-15 * pi / 180])
                ubg.append([15 * pi / 180])

            # # optional constraints
            # if self.fix_mach:
            #     for k in range(self.nodes - 1):
            #         g.append(U[k + 1][0] - U[k][0])
            #         lbg.append([0])
            #         ubg.append([0])

            # if self.fix_alt:
            #     for k in range(self.nodes):
            #         g.append(U[k][1])
            #         lbg.append([0])
            #         ubg.append([0])

            # if self.fix_track:
            #     for k in range(self.nodes - 1):
            #         g.append(U[k + 1][2] - U[k][2])
            #         lbg.append([0])
            #         ubg.append([0])

            # if not self.allow_descent:
            #     for k in range(self.nodes):
            #         g.append(U[k][1])
            #         lbg.append([0])
            #         ubg.append([ca.inf])

        # Separation constraint
        Rxy = 6.57 * oc.aero.nm
        Rz = 1541 * oc.aero.ft
        if len(self.scenarios) == 2:
            # for k in range(self.scenarios[0]["nodes"]):
            #     for k2 in range(self.scenarios[1]["nodes"]):
            #         v_tas2 = oc.aero.mach2tas(
            #             U[1][k2][0], X[1][k2][2], dT=self.dT
            #         )  # m/s
            #         vs2 = U[1][k2][1]  # m/s
            #         psi2 = U[1][k2][2]  # rad
            #         ts1 = self.scenarios[0]["tstart"] + X[0][k][4]
            #         ts2 = self.scenarios[1]["tstart"] + X[1][k2][4]
            #         dist_x = v_tas2 * ca.sin(psi2) * (ts1 - ts2) - (
            #             X[0][k][0] - X[1][k2][0]
            #         )
            #         dist_y = v_tas2 * ca.cos(psi2) * (ts1 - ts2) - (
            #             X[0][k][1] - X[1][k2][1]
            #         )
            #         dist_z = vs2 * (ts1 - ts2) - (X[0][k][2] - X[1][k2][2])
            #         ellipsoid = (
            #             (dist_x / Rxy) ** 2 + (dist_y / Rxy) ** 2 + (dist_z / Rz) ** 2
            #         )
            #         g.append(ellipsoid)
            #         lbg.append([1.2])
            #         ubg.append([ca.inf])

            t_start_global = min(sc["tstart"] for sc in self.scenarios)
            t_end_global = max(
                sc["tstart"] + sc["range"] / 200 for sc in self.scenarios
            )
            min_nodes = min(min(sc["nodes"] for sc in self.scenarios), 20)
            t_sync = np.linspace(t_start_global, t_end_global, 2 * min_nodes)

            for t in t_sync:
                Xsync = []

                for isc, sc in enumerate(self.scenarios):

                    x_t = self.interpolate_state_global(
                        X[isc],
                        Xc_store[isc],
                        sc["dt"],
                        sc["tstart"],
                        t,
                        sc["nodes"],
                    )

                    Xsync.append(x_t)

                # ---- separation constraint (2 scenarios) ----
                dist_x = Xsync[0][0] - Xsync[1][0]
                dist_y = Xsync[0][1] - Xsync[1][1]
                dist_z = Xsync[0][2] - Xsync[1][2]

                ellipsoid = (
                    (dist_x / Rxy) ** 2 + (dist_y / Rxy) ** 2 + (dist_z / Rz) ** 2
                )

                g.append(ellipsoid)
                lbg.append([1])
                ubg.append([ca.inf])

        # # add fuel constraint
        # g.append(X[0][3] - X[-1][3])
        # lbg.append([0])
        # ubg.append([self.fuel_max])

        # if customized_max_fuel is not None:
        #     g.append(X[0][3] - X[-1][3] - customized_max_fuel)
        #     lbg.append([-ca.inf])
        #     ubg.append([0])

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

        w_opt = self.solution["x"]

        trajectories = []

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
        # final timestep
        # ts_final = self.solution["x"][-1].full().item()
        # print(X)
        # print("self.solution ", self.solution)
        # print("ts_final: ", ts_final)
        # # Function to get x and u from w
        # output = ca.Function("output", [w], [X, U], ["w"], ["x", "u"])
        # x_opt, u_opt = output(self.solution["x"])

        # # --- extract trajectories for ALL scenarios ---
        # dfs = []

        # for isc, sc in enumerate(self.scenarios):
        #     df_i = self.to_trajectory(ts_final[isc], x_opt[isc], u_opt[isc])
        #     df_i["scenario"] = isc
        #     dfs.append(df_i)

        # self.trajectories = dfs

        # df_copy = df.copy()

        # check if the optimizer has failed
        if not self.solver.stats()["success"]:
            warnings.warn("flight might be infeasible.")

        if df.altitude.max() < 5000:
            warnings.warn("max altitude < 5000 ft, optimization seems to have failed.")
            df = None

        if df is not None:
            final_mass = df.mass.iloc[-1]

            if final_mass < self.oew:
                warnings.warn("final mass condition violated (smaller than OEW).")
                df = None
