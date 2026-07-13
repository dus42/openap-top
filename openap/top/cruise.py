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
        sc["u_0_lb"] = [0.5, -500 * fpm, psi - pi]
        sc["u_0_ub"] = [sc["mach_max"], 500 * fpm, psi + pi]

        # Control final - lower and upper bounds
        sc["u_f_lb"] = [0.5, -500 * fpm, psi - pi]
        sc["u_f_ub"] = [sc["mach_max"], 500 * fpm, psi + pi]

        # Control - Lower and upper bound
        sc["u_lb"] = [0.5, -500 * fpm, psi - pi]
        sc["u_ub"] = [sc["mach_max"], 500 * fpm, psi + pi]

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
                # accept a list (one per scenario) or a single DataFrame
                ig = (
                    initial_guess[isc]
                    if isinstance(initial_guess, (list, tuple))
                    else initial_guess
                )
                sc["x_guess"] = self.initial_guess(sc, flight=ig)
            else:
                sc["x_guess"] = self.initial_guess(sc)

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
            w0.append([sc["range"] / 200])

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
                lbg.append([-1])
                ubg.append([1])


            # smooth heading change
            for k in range(nodes - 1):
                g.append(U[isc][k + 1][2] - U[isc][k][2])
                lbg.append([-15 * pi / 180])
                ubg.append([15 * pi / 180])

            # optional constraints
            if self.fix_mach:
                for k in range(nodes - 1):
                    g.append(U[isc][k + 1][0] - U[isc][k][0])
                    lbg.append([0])
                    ubg.append([0])

            if self.fix_alt:
                for k in range(nodes):
                    g.append(U[isc][k][1])
                    lbg.append([0])
                    ubg.append([0])

            if self.fix_track:
                for k in range(nodes - 1):
                    g.append(U[isc][k + 1][2] - U[isc][k][2])
                    lbg.append([0])
                    ubg.append([0])

            if not self.allow_descent:
                for k in range(nodes):
                    g.append(U[isc][k][1])
                    lbg.append([0])
                    ubg.append([ca.inf])

        # # Separation constraint
        if len(self.scenarios) > 1 and self.conflict:
            Rxy = 5.0 * oc.aero.nm
            Rz = 1000 * oc.aero.ft

            guess = kwargs.get("initial_guess", None)
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

        # check if the optimizer has failed
        # if not self.solver.stats()["success"]:
        #     warnings.warn("flight might be infeasible.")

        # # Validate each trajectory
        # for isc, df in enumerate(trajectories):
        #     if df.altitude.max() < 5000:
        #         warnings.warn(
        #             f"Scenario {isc}: max altitude < 5000 ft, optimization seems to have failed."
        #         )

        #     final_mass = df.mass.iloc[-1]
        #     sc = self.scenarios[isc]
        #     if final_mass < sc["oew"]:
        #         warnings.warn(
        #             f"Scenario {isc}: final mass condition violated (smaller than OEW)."
        #         )

        # return trajectories
