# Imports
import json2html
import io
import base64
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import RK45, LSODA, simps
from dataclasses import dataclass, InitVar, field, fields, asdict
from numpy.polynomial import polynomial as P
from abc import abstractmethod, ABC
import json
import os

# All values are put in SI units


# Controller Actions


class Action(ABC):
    "An action can be from P(Proportional), I(Integral) or D(Derivative)"

    def __init__(self, prop_const, delay_time=0):
        """K: The proportionality constant
        delay_time: The time from which action should start to take place
        """

        self.K = prop_const
        self.delay_time = delay_time

    @abstractmethod
    def take_action(self, errors, times):
        pass


class PAction(Action):
    """Implementation of proportional action"""

    def take_action(self, errors, times):
        if self.delay_time <= times[-1]:
            return self.K * errors[-1]
        else:
            return 0


class IAction(Action):
    """Implementation of  integral action"""

    def take_action(self, errors, times):
        if self.delay_time <= times[-1]:
            return self.K * simps(errors, times)
        else:
            return 0


class DAction(Action):
    """Implementation of derivative action"""

    def take_action(self, errors, times):
        if self.delay_time <= times[-1]:
            return self.K * np.gradient(errors, times)[-1]
        else:
            return 0


class Controller:

    """Implementation of a Controller"""

    def __init__(self, set_point, actions):
        self.errors = []
        self.times = []

        self.set_point = set_point
        assert bool(actions) == True, "Atleast one action is required"
        self.actions = actions

    def _set_errors(self, error):
        self.errors.append(error)

    def _set_times(self, time):
        self.times.append(time)

    def get_response(self, current_time, current_val):
        """"Get the response based on the current value and currentn time"""
        self._set_times(current_time)
        self._set_errors(self.set_point - current_val)

        total_action = 0
        for action in self.actions:
            total_action += action.take_action(self.errors, self.times)
        return total_action


# Ode Solving
@dataclass
class InstantaneousData:
    """"Data structure prepared from the values in the derivates"""

    y: InitVar[np.ndarray]
    t: InitVar[float]

    # liquid
    VL: float = field(init=False)
    Fo: float = field(init=False)
    T: float = field(init=False)
    P: float = field(init=False)
    HL: float = field(init=False)
    UL: float = field(init=False)
    Uv: float = field(init=False)

    # vapour
    Vv: float = field(init=False)
    rho_v: float = field(init=False)
    Tv: float = field(init=False)
    Pv: float = field(init=False)
    Hv: float = field(init=False)

    # others
    Q: float = field(init=False)
    Wv: float = field(init=False)

    def __post_init__(self, t, y):
        # liquid

        self.time = t
        self.VL = y[0] / rho_o
        temp = Fo_steady + functions["devFo"](t=t, VL=self.VL)
        self.Fo = temp if temp >= 0 else 0
        self.T = y[1] / (self.VL * Cp * rho_o)
        self.P = functions["P"](T=self.T)
        self.HL = functions["HL"](T=self.T)
        self.UL = functions["HL"](T=self.T)
        self.Uv = functions["Uv"](T=self.T)

        # vapour
        self.Vv = V_chamber - self.VL
        self.rho_v = y[2] / self.Vv
        self.Tv = y[3] / (self.Vv * self.rho_v * Cp)
        self.Pv = functions["Pv"](rho_v=self.rho_v, T=self.Tv)
        self.Hv = Cp * self.Tv

        self.Wv = functions["Wv"](P=self.P, Pv=self.Pv)
        # others
        temp = Q_steady + functions["devQ"](
            t=t, P=self.P + self.Pv
        )  # P or Pv or P + Pv?????
        self.Q = temp if temp >= 0 else 0


def derivative(t, y):
    """Calculate the derivative for the solver algorithm"""

    assert y.ndim == 1, "Dimension is %s" % y.ndim
    data = InstantaneousData(t, y)

    return np.array(
        [
            rho_o * data.Fo - data.Wv,
            rho_o * data.Fo * ho - data.Wv * data.HL + data.Q,
            data.Wv - data.rho_v * Fv,
            data.Wv * data.HL - data.rho_v * Fv * data.Hv,
        ]
    )


functions = {
    "P": lambda T: 10 ** (A - B / (T + C)) * (10 ** 5),
    "Wv": lambda P, Pv: Km * (P - Pv),
    "Pv": lambda rho_v, T: (rho_v * R * T) / M,
    "devQ": lambda t, P: pressure_controller.get_response(t, P),
    "devFo": lambda t, VL: level_controller.get_response(t, VL),
    "HL": lambda T: latent_heat + T * Cp,
    "UL": lambda T: T * Cp,
    "Uv": lambda T: T * Cp,
}


def make_action(k, v):
    return {"P": PAction, "I": IAction, "D": DAction}[k](
        prop_const=v["prop_constant"], delay_time=v["delay_time"]
    )


def fill_dataframe(output):
    while output.status == "running":
        output.step()
        temp = {
            "time": output.t,
            **asdict(InstantaneousData(output.t, output.y)),
        }
        yield temp


# thermodynamic constants
R = 8.314  # J⋅ /K⋅ /mol


# Compound properties
Cp = 1.68 * 1000  # J / kg  / K
M = 44.1 / 1000  # kg / mol
latent_heat = 428 * 1000  # J / kg

# Antoine Equation
# Source: https://webbook.nist.gov/cgi/cbook.cgi?ID=C74986&Mask=4&Type=ANTOINE&Plot=on#ANTOINE
A = 3.98292
B = 819.296  # K
C = -24.417  # K


with open("input_data.json") as f:
    data_input = json.load(f)


for index, dict_item in data_input.items():

    ############Set Values############

    level_controller = Controller(
        set_point=dict_item["controllers"]["level"]["set_point"],
        actions=(
            [
                make_action(k, v)
                for k, v in dict_item["controllers"]["level"][
                    "actions"
                ].items()
            ]
        ),
    )

    pressure_controller = Controller(
        set_point=dict_item["controllers"]["pressure"]["set_point"],
        actions=(
            [
                make_action(k, v)
                for k, v in dict_item["controllers"]["pressure"][
                    "actions"
                ].items()
            ]
        ),
    )

    # pseudo-mass transfer coefficient
    Km = dict_item["Km"]

    # RK 45 parameters
    t_bound = dict_item["t_bound"]

    # Chamber properties
    V_chamber = dict_item["V_chamber"]  # m^3

    # Inlet
    rho_o = dict_item["rho_o"]  # kg / m^ 3
    To = dict_item["To"]  # K
    ho = Cp * To

    # Outlet
    Fv = dict_item["Fv"]  # m^3/s

    # Steady state values
    Q_steady = dict_item["Q_steady"]  # J
    Fo_steady = dict_item["Fo_steady"]  # m^3/s

    VL_frac = dict_item["VL_frac"]

    ###############################################################################
    ################################################################################
    ################################################################################

    # t  = 0 , intial values of variables based on steady state
    VL = VL_frac * V_chamber  # m^3
    Vv = V_chamber - VL
    rho_v = rho_o * Fo_steady / Fv
    rho_o_VL = rho_o * VL
    rho_VL_Cp_T = rho_o * VL * Cp * To
    Vv_rho_v = Vv * rho_v
    rho_v_Vv_Cp_T = rho_v * Vv * Cp * To

    output = {"LSODA": LSODA, "RK45": RK45}[dict_item["method"]](
        fun=derivative,
        t0=0,
        y0=np.array([rho_o_VL, rho_VL_Cp_T, Vv_rho_v, rho_v_Vv_Cp_T]),
        t_bound=t_bound,
    )

    df = pd.DataFrame(data=[*fill_dataframe(output)])

    ncols = 3
    nrows = 5
    fig, ax = plt.subplots(nrows, ncols, figsize=(27, 16))

    for i, v in enumerate(list(df.columns)[1:]):
        ax[i % nrows, i % ncols].plot(df["time"], df[v])
        ax[i % nrows, i % ncols].set_xlabel("time")
        ax[i % nrows, i % ncols].set_ylabel(v)

    fig.delaxes(ax[nrows - 1][ncols - 1])
    name = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{index}'

    image_bytes = io.BytesIO()

    if not os.path.exists("datas"):
        os.mkdir("datas")

    plt.savefig(f"datas/{name}.png", format="png", bbox_inches="tight")
    plt.savefig(image_bytes, format="png", bbox_inches="tight")

    image_bytes = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

    plt.close()

    with open("datas/" + name + ".htm", "w", encoding="utf-8") as f:
        f.write(
            f"""   
                <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta http-equiv="X-UA-Compatible" content="IE=edge">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{name}</title>
            </head>
            <body>
                <h2>Parameters used:</h2>
                
<ul>R = {R}   J⋅ /K⋅ /mol


<li>Cp = {Cp}   J / kg  / K</li>
<li>M = {M}   kg / mol</li>
<li>Latent Heat = {latent_heat}   J / kg</li>

</ul>

  <h2>Parameters used(Json Data):</h2>
{json2html.json2html.convert(json = dict_item)}


<a href="https://webbook.nist.gov/cgi/cbook.cgi?ID=C74986&Mask=4&Type=ANTOINE&Plot=on#ANTOINE"><h3> Antoine Equation:</h3></a>
<ul>
<li>A = {A}</li>
<li>B = {B}   K</li>
<li>C = {C}   K</li>
</ul>
           
 <h2>Simulation Data:</h2>               
                {df.to_html()}
         
           
 <h2>Plots:</h2>               
         
            
            <img align="left" src="data:image/png;base64,{image_bytes}">   
            </body>
            </html>
                
                """
        )
