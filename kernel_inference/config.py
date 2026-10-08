"""One model configuration: everything that defines a fit, with its default."""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

Kernel = Literal["bounded_confidence", "sigmoidal_bounded_confidence", "simplified_degroot",
                 "rzb", "null", "legendre", "random_fourier"]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fit: str                                  # <noise>_<arm>/<panel>/<kernel>, names the results
    panel: str
    kernel_func: Kernel
    phi_init: list[list[float]]               # one restart per entry
    rff_seed: list[int] = [42]                # random-feature draws, one restart each

    cv_num_folds: int = 5
    cv_fold_index: list[int] = [0, 1, 2, 3, 4]
    use_async_em: bool = False                # one actor per time step (cmv, markets, spinos)
    no_opinion_value: Optional[float] = None  # a sentinel opinion that is not an opinion
    dequantize: bool = False
    grid_h: Optional[float] = None            # width of the grid opinions were recorded on
    representation: Literal["pairwise", "neighbour_mean_field"] = "pairwise"

    noise_model: Literal["gaussian", "laplace"] = "gaussian"
    sigma_init: Optional[float] = None        # None: the panel's own noise scale
    lmbda_init: Optional[float] = None        # None: 1
    infer_sigma: bool = True
    infer_lmbda: bool = True
    lmbda_min: float = 0.0
    infer_pi: bool = False
    init_pi: float = 1.0
    pi_min: float = 1e-3
    pi_max: float = 1.0 - 1e-3
    decoupled_scale: bool = False             # the null component has its own scale b_null
    init_b_null: Optional[float] = None
    b_null_min: Optional[float] = None
    infer_mu_null: bool = False

    kernel_positive_phi_transform: bool = False
    kernel_phi_floor: float = 0.0
    degroot_unit_gain: bool = False
    legendre_num_basis: int = 9
    legendre_max_distance: float = 1.0
    legendre_degree_penalty: float = 0.02
    legendre_unit_gain_at_zero: bool = False
    rff_gamma: float = 2.0
    rff_num_features: int = 256
    rff_unit_gain_at_zero: bool = False

    optimization_method: Literal["gradient_descent", "closed_form", "bisection"] = "gradient_descent"
    optimizer_class: str = "adagrad"
    gd_rate: float = 0.1
    max_gd_iter: int = 250
    max_em_iter: int = 100
    ll_stop_patience: int = 3
    ll_stopping_criterion: float = 1e-4
    bisection_max_iter: int = 100
    bisection_max_distance: Optional[float] = None

    e_step_chunk_size: Optional[int] = None
    events_dtype: Literal["float64", "float32"] = "float64"
    kernel_loss_checkpoint: bool = False
