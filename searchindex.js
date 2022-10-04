Search.setIndex({"docnames": ["add", "api", "auto_examples/index", "auto_examples/plot_compare_time", "auto_examples/plot_lasso_vs_weighted", "auto_examples/plot_logreg_various_penalties", "auto_examples/plot_pen_prox", "auto_examples/plot_sparse_recovery", "auto_examples/plot_zero_weights_lasso", "auto_examples/sg_execution_times", "generated/skglm.ElasticNet", "generated/skglm.GeneralizedLinearEstimator", "generated/skglm.Lasso", "generated/skglm.LinearSVC", "generated/skglm.MCPRegression", "generated/skglm.MultiTaskLasso", "generated/skglm.SparseLogisticRegression", "generated/skglm.WeightedLasso", "generated/skglm.datafits.Huber", "generated/skglm.datafits.Logistic", "generated/skglm.datafits.Quadratic", "generated/skglm.datafits.QuadraticGroup", "generated/skglm.datafits.QuadraticSVC", "generated/skglm.penalties.IndicatorBox", "generated/skglm.penalties.L0_5", "generated/skglm.penalties.L1", "generated/skglm.penalties.L1_plus_L2", "generated/skglm.penalties.L2_3", "generated/skglm.penalties.MCPenalty", "generated/skglm.penalties.WeightedGroupL2", "generated/skglm.penalties.WeightedL1", "generated/skglm.solvers.AndersonCD", "generated/skglm.solvers.GramCD", "generated/skglm.solvers.GroupBCD", "generated/skglm.solvers.MultiTaskBCD", "generated/skglm.solvers.ProxNewton", "index"], "filenames": ["add.rst", "api.rst", "auto_examples/index.rst", "auto_examples/plot_compare_time.rst", "auto_examples/plot_lasso_vs_weighted.rst", "auto_examples/plot_logreg_various_penalties.rst", "auto_examples/plot_pen_prox.rst", "auto_examples/plot_sparse_recovery.rst", "auto_examples/plot_zero_weights_lasso.rst", "auto_examples/sg_execution_times.rst", "generated/skglm.ElasticNet.rst", "generated/skglm.GeneralizedLinearEstimator.rst", "generated/skglm.Lasso.rst", "generated/skglm.LinearSVC.rst", "generated/skglm.MCPRegression.rst", "generated/skglm.MultiTaskLasso.rst", "generated/skglm.SparseLogisticRegression.rst", "generated/skglm.WeightedLasso.rst", "generated/skglm.datafits.Huber.rst", "generated/skglm.datafits.Logistic.rst", "generated/skglm.datafits.Quadratic.rst", "generated/skglm.datafits.QuadraticGroup.rst", "generated/skglm.datafits.QuadraticSVC.rst", "generated/skglm.penalties.IndicatorBox.rst", "generated/skglm.penalties.L0_5.rst", "generated/skglm.penalties.L1.rst", "generated/skglm.penalties.L1_plus_L2.rst", "generated/skglm.penalties.L2_3.rst", "generated/skglm.penalties.MCPenalty.rst", "generated/skglm.penalties.WeightedGroupL2.rst", "generated/skglm.penalties.WeightedL1.rst", "generated/skglm.solvers.AndersonCD.rst", "generated/skglm.solvers.GramCD.rst", "generated/skglm.solvers.GroupBCD.rst", "generated/skglm.solvers.MultiTaskBCD.rst", "generated/skglm.solvers.ProxNewton.rst", "index.rst"], "titles": ["How to add a custom penalty", "API Documentation", "Examples Gallery", "Timing comparison with scikit-learn for Lasso", "Comparison of Lasso and Weighted Lasso", "Logistic regression with Elastic net and minimax concave penalties", "Value and proximal operators of penalties", "Sparse recovery with non-convex penalties", "Weighted Lasso with some zero weights", "Computation times", "skglm.ElasticNet", "skglm.GeneralizedLinearEstimator", "skglm.Lasso", "skglm.LinearSVC", "skglm.MCPRegression", "skglm.MultiTaskLasso", "skglm.SparseLogisticRegression", "skglm.WeightedLasso", "skglm.datafits.Huber", "skglm.datafits.Logistic", "skglm.datafits.Quadratic", "skglm.datafits.QuadraticGroup", "skglm.datafits.QuadraticSVC", "skglm.penalties.IndicatorBox", "skglm.penalties.L0_5", "skglm.penalties.L1", "skglm.penalties.L1_plus_L2", "skglm.penalties.L2_3", "skglm.penalties.MCPenalty", "skglm.penalties.WeightedGroupL2", "skglm.penalties.WeightedL1", "skglm.solvers.AndersonCD", "skglm.solvers.GramCD", "skglm.solvers.GroupBCD", "skglm.solvers.MultiTaskBCD", "skglm.solvers.ProxNewton", "skglm"], "terms": {"With": [0, 28], "skglm": [0, 3, 4, 5, 6, 7, 8], "you": [0, 36], "can": [0, 10, 12, 13, 14, 15, 17, 32, 36], "solv": [0, 3, 10, 11, 12, 13, 14, 15, 16, 17, 36], "ani": [0, 5, 36], "gener": [0, 2, 3, 4, 5, 6, 7, 8, 11, 35, 36], "linear": [0, 11, 14, 35, 36], "model": [0, 3, 31, 33, 35, 36], "arbitrari": [0, 36], "smooth": [0, 13], "proxim": [0, 2, 9], "defin": [0, 7], "two": [0, 5, 21, 29], "class": [0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], "thei": [0, 5], "pass": [0, 5], "generalizedlinearestim": [0, 5], "clf": [0, 8], "mydatafit": 0, "mypenalti": 0, "A": [0, 16, 22, 31, 32, 35, 36], "i": [0, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 23, 31, 32, 33, 35, 36], "jitclass": 0, "which": [0, 7, 32], "must": [0, 13, 16], "inherit": 0, "from": [0, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 17], "basepenalti": [0, 11], "base": [0, 12, 17, 32], "subclass": 0, "abstractmethod": 0, "def": [0, 3], "get_spec": 0, "self": 0, "specifi": [0, 10, 11, 12, 13, 14, 15, 16, 17], "numba": [0, 36], "type": 0, "attribut": [0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 29, 32, 33], "return": [0, 3], "spec": 0, "tupl": 0, "attribute_nam": 0, "dtype": 0, "compil": [0, 3, 11, 36], "params_to_dict": 0, "get": 0, "paramet": [0, 5, 10, 11, 12, 13, 14, 15, 16, 17, 29], "initi": [0, 10, 11, 12, 13, 14, 15, 16, 17, 32, 33], "an": [0, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32], "instanc": [0, 11], "dict_of_param": 0, "dict": 0, "The": [0, 5, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 29, 31], "instanti": 0, "object": [0, 3, 10, 12, 13, 14, 15, 16, 17], "valu": [0, 2, 7, 9, 28, 32, 33], "w": [0, 3, 7, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 32], "vector": [0, 10, 12, 13, 14, 15, 17, 32, 33], "prox_1d": [0, 6], "stepsiz": 0, "j": [0, 7, 28, 35], "oper": [0, 2, 9], "featur": [0, 4, 8, 14, 16, 17, 31, 32, 35, 36], "subdiff_dist": 0, "grad": 0, "distanc": 0, "neg": 0, "gradient": [0, 18, 19, 20, 22, 32], "subdifferenti": 0, "arrai": [0, 6, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 29, 32, 33], "shape": [0, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 29, 32, 33], "n_featur": [0, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 28, 29, 32, 33], "coeffici": [0, 5, 16, 32, 33], "0": [0, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 23, 24, 28, 31, 32, 33, 34, 35], "restrict": [0, 31], "ws_size": 0, "indic": [0, 21, 23, 29], "work": [0, 10, 12, 13, 14, 15, 17, 31, 33, 35, 36], "set": [0, 5, 10, 12, 13, 14, 15, 16, 17, 23, 31, 32, 33, 35], "is_pen": 0, "binari": [0, 3, 16], "mask": 0, "penal": [0, 4, 8, 14], "generalized_support": 0, "r": [0, 7, 8], "true": [0, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 31, 32, 34, 35], "support": [0, 17, 36], "To": [0, 13, 36], "implement": [0, 36], "your": 0, "own": 0, "onli": [0, 32], "need": 0, "new": 0, "its": 0, "kkt": 0, "violat": 0, "ar": [0, 17, 36], "comput": [0, 7, 8, 20, 32], "For": [0, 10, 14], "exampl": [0, 3, 4, 5, 6, 7, 8], "l1": [0, 6, 7, 10, 11, 12, 17, 26, 30, 31, 36], "__init__": [0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], "alpha": [0, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 17, 23, 24, 25, 26, 27, 28, 29, 30], "float64": 0, "np": [0, 3, 4, 5, 6, 7, 8, 14], "sum": [0, 3], "ab": [0, 3, 4, 8, 28, 31, 35], "soft": [0, 14], "threshold": [0, 14], "st": 0, "zeros_lik": [0, 7], "idx": [0, 7], "enumer": [0, 7], "grad_j": 0, "max": [0, 3, 4, 7, 8, 13], "els": 0, "sign": [0, 5], "ones": 0, "bool_": 0, "non": [0, 2, 9, 13, 24, 27, 28, 36], "zero": [0, 2, 4, 7, 9, 32, 33], "alpha_max": [0, 4, 7, 8], "gradient0": 0, "solut": [0, 7, 10, 12, 13, 14, 15, 16, 17], "basedatafit": [0, 11], "x": [0, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 28, 32], "y": [0, 3, 4, 5, 7, 8, 10, 12, 13, 14, 15, 17, 20, 21, 22, 32], "pre": [0, 20], "befor": 0, "fit": [0, 3, 4, 5, 8, 10, 11, 12, 13, 14, 15, 16, 17, 31, 33], "n_sampl": [0, 4, 5, 7, 8, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 23, 32, 33], "design": 0, "matrix": [0, 10, 11, 12, 13, 14, 15, 17, 32], "target": 0, "initialize_spars": 0, "x_data": 0, "x_indptr": 0, "x_indic": 0, "when": [0, 4, 10, 11, 12, 13, 14, 15, 16, 17, 32], "spars": [0, 2, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 28, 31, 35, 36], "n_element": 0, "data": [0, 5, 7], "csc": 0, "1": [0, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 28, 29, 31, 32, 35], "indptr": 0, "xw": [0, 32], "array_lik": 0, "float": [0, 10, 12, 13, 14, 15, 16, 17, 29, 31, 32, 33, 35], "gradient_scalar": 0, "respect": 0, "th": 0, "coordin": [0, 11, 14, 31, 32, 33, 34, 36], "int": [0, 3, 7, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33, 35], "evalu": [0, 20], "gradient_scalar_spars": 0, "dimens": 0, "along": 0, "method": [0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], "declar": 0, "see": [0, 14], "quadrat": [0, 7, 11, 21, 22, 32], "read": [0, 18, 19, 20, 21, 22, 29], "2": [0, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 27, 28, 31, 32, 35], "2_2": [0, 10, 12, 13, 14, 15, 17, 20, 21, 22], "xty": [0, 20], "quantiti": [0, 20], "us": [0, 4, 5, 8, 10, 12, 13, 14, 15, 17, 20, 21, 22, 31, 32, 33, 36], "dure": [0, 20], "equal": [0, 14, 17, 19, 20], "t": [0, 3, 4, 7, 8, 13, 16, 20, 22, 32, 35], "lipschitz": [0, 18, 19, 20, 21, 22], "coordinatewis": [0, 18, 19, 20, 22], "constant": [0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], "norm": [0, 3, 4, 5, 7, 12, 17, 19, 20, 24, 27, 32, 36], "axi": [0, 4, 5, 7, 19, 20], "note": [0, 17, 23, 28], "jit": [0, 11], "time": [0, 2, 4, 5, 6, 7, 8, 36], "thi": [0, 8, 11, 13, 32, 36], "allow": [0, 8], "faster": [0, 31], "rang": [0, 3, 7], "len": [0, 3, 4, 5, 7, 8], "nrm2": 0, "xjtxw": 0, "full_grad_spars": 0, "intercept_update_step": 0, "mean": [0, 16], "penalti": [2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 17, 21, 32, 36], "comparison": [2, 9], "lasso": [2, 7, 9, 10, 14, 17, 36], "weight": [2, 6, 9, 12, 17, 29, 30, 36], "some": [2, 4, 6, 9], "logist": [2, 9, 16], "regress": [2, 9, 11, 14, 16], "elast": [2, 9, 10], "net": [2, 9, 10], "minimax": [2, 9, 28], "concav": [2, 9, 28], "scikit": [2, 9, 36], "learn": [2, 9, 36], "recoveri": [2, 9], "convex": [2, 9, 23, 24, 27, 28, 36], "download": [2, 3, 4, 5, 6, 7, 8], "all": [2, 7, 8, 36], "python": [2, 3, 4, 5, 6, 7, 8, 36], "sourc": [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], "code": [2, 3, 4, 5, 6, 7, 8, 31, 35, 36], "auto_examples_python": 2, "zip": [2, 3], "jupyt": [2, 3, 4, 5, 6, 7, 8], "notebook": [2, 3, 4, 5, 6, 7, 8], "auto_examples_jupyt": 2, "sphinx": [2, 3, 4, 5, 6, 7, 8], "click": [3, 4, 5, 6, 7, 8], "here": [3, 4, 5, 6, 7, 8], "full": [3, 4, 5, 6, 7, 8], "compar": 3, "larg": [3, 8], "scale": [3, 35], "problem": [3, 11, 13, 16, 31, 33, 34, 36], "file_s": 3, "00": [3, 9], "26": 3, "8m": 3, "b": [3, 35], "": [3, 5, 7, 35], "8": [3, 6], "19k": 3, "10": [3, 4, 8, 10, 12, 13, 14, 15, 17, 31, 33, 34, 35], "38": [3, 9], "41": 3, "9kb": 3, "65": 3, "5k": 3, "02": [3, 7], "21": [3, 15], "189kb": 3, "90": 3, "1k": 3, "47": 3, "159kb": 3, "205k": 3, "01": [3, 9], "326kb": 3, "4": [3, 4, 6, 7, 19, 32, 33, 35], "434k": 3, "42": 3, "627kb": 3, "3": [3, 4, 5, 6, 7, 8, 14, 27], "9": 3, "893k": 3, "20mb": 3, "7": [3, 4, 6, 7], "81m": 3, "32mb": 3, "14": 3, "5": [3, 4, 7, 8, 10, 24, 32], "65m": 3, "05": 3, "50mb": 3, "19": 3, "22m": 3, "03": [3, 9], "58mb": 3, "25": 3, "6": [3, 5, 7], "79m": 3, "31mb": 3, "31": 3, "36m": 3, "82mb": 3, "37": 3, "94m": 3, "17mb": 3, "43": 3, "11": [3, 5, 9], "5m": 3, "40mb": 3, "49": 3, "13": [3, 7, 9], "1m": 3, "57mb": 3, "55": 3, "7m": 3, "70mb": 3, "61": [3, 5], "16": 3, "2m": 3, "77mb": 3, "66": 3, "17": 3, "83mb": 3, "72": 3, "4m": 3, "88mb": 3, "78": 3, "20": [3, 16, 35], "9m": 3, "89mb": 3, "84": 3, "22": 3, "24": 3, "04": 3, "91mb": 3, "96": 3, "100": [3, 5, 8, 32, 33, 34], "35mb": 3, "import": [3, 4, 5, 6, 7, 8, 36], "warn": 3, "numpi": [3, 4, 5, 6, 7, 8], "linalg": [3, 4, 5, 7], "matplotlib": [3, 4, 5, 6, 7, 8], "pyplot": [3, 4, 5, 6, 7, 8], "plt": [3, 4, 5, 6, 7, 8], "libsvmdata": 3, "fetch_libsvm": 3, "sklearn": [3, 5, 7, 36], "except": 3, "convergencewarn": 3, "linear_model": 3, "lasso_sklearn": 3, "elasticnet": [3, 26], "enet_sklearn": 3, "filterwarn": 3, "ignor": 3, "categori": 3, "compute_obj": 3, "l1_ratio": [3, 5, 6, 10, 26], "loss": [3, 13], "news20": 3, "dict_sklearn": 3, "fit_intercept": [3, 4, 7, 8, 10, 12, 13, 14, 15, 16, 17, 31, 32, 33, 34, 35], "fals": [3, 4, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 31, 32, 33, 34, 35], "tol": [3, 10, 12, 13, 14, 15, 16, 17, 31, 32, 33, 34, 35], "1e": [3, 7, 32, 33, 34, 35], "12": [3, 7], "enet": 3, "dict_our": 3, "fig": [3, 4, 6, 7, 8], "axarr": [3, 4, 6, 7, 8], "subplot": [3, 4, 6, 7, 8], "constrained_layout": [3, 6, 7, 8], "ax": 3, "pobj_dict": 3, "list": [3, 16], "u": 3, "time_dict": 3, "remov": [3, 5], "max_it": [3, 10, 12, 13, 14, 15, 16, 17, 31, 32, 33, 34, 35], "10_000": 3, "w_star": [3, 5], "coef_": [3, 4, 5, 8, 10, 11, 12, 13, 14, 15, 16, 17], "pobj_star": 3, "n_iter_sklearn": 3, "uniqu": 3, "geomspac": [3, 7], "50": [3, 4, 5, 8, 10, 12, 13, 14, 15, 17, 31], "num": [3, 6, 7], "15": 3, "astyp": 3, "t_start": 3, "w_sklearn": 3, "append": 3, "n_iter_u": 3, "semilogi": 3, "label": [3, 5, 6, 7, 16, 19], "set_ylim": [3, 7], "set_titl": [3, 4, 6, 8], "legend": [3, 5, 6, 7], "set_ylabel": [3, 7], "suboptim": 3, "set_xlabel": [3, 7, 8], "show": [3, 4, 5, 6, 7, 8], "block": [3, 4, 6, 7, 8, 31, 33, 34], "total": [3, 4, 5, 6, 7, 8, 9], "run": [3, 4, 5, 6, 7, 8, 11], "script": [3, 4, 5, 6, 7, 8], "minut": [3, 4, 5, 6, 7, 8], "898": [3, 9], "second": [3, 4, 5, 6, 7, 8, 36], "plot_compare_tim": [3, 9], "py": [3, 4, 5, 6, 7, 8, 9], "ipynb": [3, 4, 5, 6, 7, 8], "galleri": [3, 4, 5, 6, 7, 8], "illustr": [4, 5, 6, 7], "normal": 4, "author": [4, 5, 6, 7, 36], "mathurin": [4, 6, 7], "massia": [4, 6, 7, 31, 35, 36], "quentin": [4, 7], "bertrand": [4, 7, 31, 36], "weightedlasso": [4, 8, 12], "util": [4, 5, 7, 8], "make_correlated_data": [4, 5, 7, 8], "30": [4, 7], "_": [4, 5, 15], "random_st": [4, 5, 7, 8], "w_true": [4, 7, 8], "nnz": 4, "assum": 4, "reason": 4, "have": 4, "smaller": 4, "than": [4, 12, 14, 17], "other": 4, "nois": 4, "random": [4, 7, 8], "randn": 4, "signal": 4, "ratio": 4, "doe": 4, "select": [4, 14, 32], "small": [4, 8], "while": 4, "la": 4, "wei": 4, "sharei": [4, 7, 8], "figsiz": [4, 6, 7, 8], "stem": [4, 5, 8], "coeff": 4, "343": [4, 9], "plot_lasso_vs_weight": [4, 9], "modular": 5, "one": 5, "datafit": [5, 7, 11, 13, 32, 36], "home": 5, "circleci": 5, "project": 5, "plot_logreg_various_penalti": [5, 9], "53": 5, "matplotlibdeprecationwarn": 5, "use_line_collect": 5, "wa": 5, "deprec": 5, "minor": 5, "releas": 5, "later": 5, "If": [5, 11, 14, 17, 32, 33, 36], "follow": 5, "should": [5, 14, 32, 36], "keyword": 5, "position": 5, "m": [5, 31, 35, 36], "69": 5, "pierr": 5, "antoin": 5, "bannier": [5, 31, 36], "metric": [5, 7], "f1_score": [5, 7], "l1_plus_l2": [5, 6], "mcpenalti": [5, 6, 7], "y_ind": 5, "standard": [5, 7], "mcp": [5, 7, 14, 28], "sqrt": [5, 7], "split": 5, "train": 5, "test": 5, "x_train": 5, "y_train": 5, "x_test": [5, 7], "y_test": [5, 7], "005": 5, "gamma": [5, 6, 7, 14, 28], "clf_enet": 5, "y_pred_enet": 5, "predict": 5, "f1_score_enet": 5, "clf_mcp": 5, "y_pred_mcp": 5, "f1_score_mcp": 5, "where": [5, 18, 23, 32], "ravel": 5, "markerfmt": 5, "setp": 5, "color": [5, 7], "2ca02c": 5, "ff7f0e": 5, "bx": 5, "loc": [5, 7], "best": 5, "titl": [5, 36], "f1": [5, 7], "3f": 5, "819": [5, 9], "gmail": 6, "com": [6, 31, 35, 36], "weightedl1": 6, "scad": [6, 7], "l0_5": [6, 7], "l2_3": [6, 7], "x_rang": 6, "linspac": 6, "300": 6, "pen": [6, 14, 28], "plot": [6, 7], "__class__": 6, "__name__": 6, "537": [6, 9], "plot_pen_prox": [6, 9], "superior": 7, "perform": 7, "l05": 7, "l23": 7, "klopfenstein": [7, 31, 36], "model_select": 7, "train_test_split": 7, "mean_squared_error": 7, "solver": [7, 8, 11, 12, 17, 36], "andersoncd": [7, 11], "compiled_clon": 7, "cmap": 7, "get_cmap": 7, "tab10": 7, "simul": 7, "1000": [7, 16, 33, 35], "densiti": 7, "seed": [7, 8], "supp": 7, "choic": [7, 8], "size": [7, 10, 12, 13, 14, 15, 17, 31], "replac": [7, 8, 11, 36], "x_": 7, "y_": 7, "snr": 7, "rho": 7, "test_siz": 7, "lead": 7, "ord": 7, "inf": [7, 14], "n_alpha": 7, "estimation_error": 7, "prediction_error": 7, "l0": 7, "mse_ref": 7, "ws_strategi": [7, 10, 12, 13, 14, 15, 17, 31, 34], "fixpoint": [7, 10, 12, 13, 14, 15, 17, 31], "estim": [7, 10, 11, 12, 13, 14, 15, 16, 17, 36], "kei": 7, "print": 7, "f": [7, 18], "estimator_path": 7, "path": 7, "f1_temp": 7, "prediction_error_temp": 7, "name_estim": 7, "ell_": 7, "close": 7, "sharex": 7, "semilogx": 7, "c": [7, 13, 35, 36], "max_f1": 7, "argmax": 7, "vline": 7, "ymin": 7, "ymax": 7, "linestyl": 7, "line1": 7, "clip_on": 7, "marker": 7, "markers": 7, "min_error": 7, "argmin": 7, "lim": 7, "get_ylim": 7, "min": 7, "line2": 7, "lambda": 7, "lambda_": 7, "mathrm": 7, "score": [7, 10, 12, 13, 14, 15, 17, 31], "pred": 7, "rmse": 7, "left": 7, "out": 7, "bbox_to_anchor": 7, "lower": 7, "mode": 7, "expand": 7, "borderaxespad": 7, "ncol": 7, "804": [7, 9], "plot_sparse_recoveri": [7, 9], "demonstr": 8, "how": 8, "vanish": 8, "fast": 8, "adapt": 8, "primal": [8, 12, 13, 17], "anderson": [8, 31], "acceler": [8, 31], "dual": [8, 13, 35], "handl": [8, 11, 16], "empti": 8, "unpen": [8, 17], "first": [8, 10, 12, 13, 14, 15, 17, 31, 36], "put": 8, "last": 8, "arang": 8, "coef": 8, "neq": 8, "heavili": 8, "few": [8, 36], "lightli": 8, "mani": [8, 36], "index": 8, "490": [8, 9], "plot_zero_weights_lasso": [8, 9], "891": 9, "execut": 9, "auto_exampl": 9, "file": 9, "mb": 9, "07": 9, "max_epoch": [10, 12, 13, 14, 15, 16, 17, 31, 33, 34], "50000": [10, 12, 13, 14, 15, 17, 31, 34], "p0": [10, 12, 13, 14, 15, 17, 31, 33, 34, 35], "verbos": [10, 12, 13, 14, 15, 16, 17, 31, 32, 33, 34, 35], "0001": [10, 12, 13, 14, 15, 16, 17, 31, 32, 33, 35], "warm_start": [10, 12, 13, 14, 15, 16, 17, 31, 32, 33, 34, 35], "subdiff": [10, 12, 13, 14, 15, 17, 31, 34], "optim": [10, 11, 12, 13, 14, 15, 16, 17, 31, 35], "sum_j": [10, 12, 17], "w_j": [10, 12, 14, 17, 28], "option": [10, 11, 12, 13, 14, 15, 16, 17, 31], "strength": [10, 12, 13, 14, 15, 16, 17], "default": [10, 12, 13, 14, 15, 16, 17, 32, 33, 35, 36], "mix": 10, "l2": [10, 13, 26, 29], "combin": [10, 35, 36], "maximum": [10, 12, 13, 14, 15, 16, 17, 31, 32, 33, 35], "number": [10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33, 35], "iter": [10, 12, 13, 14, 15, 16, 17, 31, 32, 33, 35], "subproblem": [10, 11, 12, 13, 14, 15, 16, 17, 31, 35], "definit": [10, 12, 13, 14, 15, 16, 17, 31], "cd": [10, 12, 13, 14, 15, 17, 31, 36], "epoch": [10, 12, 13, 14, 15, 17, 31, 32, 33], "each": [10, 12, 13, 14, 15, 16, 17, 21, 31, 35], "bool": [10, 12, 13, 14, 15, 16, 17, 31, 32, 33, 35], "amount": [10, 12, 13, 14, 15, 16, 17, 31, 32, 33, 35], "stop": [10, 12, 13, 14, 15, 16, 17], "criterion": [10, 12, 13, 14, 15, 16, 17], "whether": [10, 12, 13, 14, 15, 16, 17, 31], "intercept": [10, 12, 13, 14, 15, 16, 17, 31], "reus": [10, 12, 13, 14, 15, 16, 17], "previou": [10, 12, 13, 14, 15, 16, 17], "call": [10, 11, 12, 13, 14, 15, 16, 17], "otherwis": [10, 12, 13, 14, 15, 16, 17, 32], "just": [10, 12, 13, 14, 15, 16, 17, 36], "eras": [10, 12, 13, 14, 15, 16, 17], "str": [10, 12, 13, 14, 15, 17], "build": [10, 12, 13, 14, 15, 17, 31], "regular": [10, 12, 13, 14, 15, 16, 17, 29], "cost": [10, 11, 12, 13, 14, 15, 17], "function": [10, 11, 12, 13, 14, 15, 16, 17, 18, 23], "formula": [10, 11, 12, 13, 14, 15, 17], "sparse_coef_": [10, 11, 12, 13, 14, 15, 17], "scipi": [10, 11, 12, 13, 14, 15, 17], "readonli": [10, 11, 12, 13, 14, 17], "properti": [10, 11, 12, 13, 14, 17], "deriv": [10, 11, 12, 13, 14, 17], "intercept_": [10, 11, 12, 13, 14, 15, 16, 17], "term": [10, 11, 12, 13, 14, 15, 16, 17], "decis": [10, 11, 12, 13, 14, 15, 16, 17], "n_iter_": [10, 11, 12, 13, 14, 15, 16, 17], "reach": [10, 11, 12, 13, 14, 15, 16, 17], "toler": [10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33, 35], "none": [11, 17, 32, 33], "take": 11, "descent": [11, 14, 31, 32, 33, 34, 36], "It": [11, 32], "classif": [11, 22], "task": [11, 22, 34], "basesolv": 11, "n_task": 11, "celer": [12, 15, 17, 35], "extrapol": [12, 17, 32, 35], "mcpregress": [12, 17], "sparser": [12, 17], "hing": 13, "sum_i": [13, 19], "y_i": [13, 16, 18, 19], "beta": 13, "e": [13, 17, 36], "we": 13, "stai": 13, "our": 13, "framework": 13, "svc": [13, 22], "w_i": 13, "ind": [13, 23], "relat": 13, "given": [13, 16], "invers": 13, "proport": 13, "strictli": 13, "posit": [13, 16, 17], "dual_": 13, "obj": 14, "more": 14, "detail": 14, "algorithm": [14, 35], "nonconvex": 14, "applic": 14, "biolog": 14, "breheni": 14, "huang": 14, "prox": [14, 16, 35], "hard": 14, "larger": 14, "multipli": 15, "l21": 15, "represent": 15, "log": [16, 19], "exp": [16, 19], "x_i": 16, "_1": 16, "outer": [16, 35], "newton": [16, 35], "classes_": 16, "ndarrai": 16, "n_class": 16, "known": 16, "classifi": 16, "Not": 16, "yet": 16, "weights_j": 17, "part": 17, "unweight": 17, "delta": 18, "sum_": [18, 28, 29], "xw_i": [18, 19], "grp_ptr": [21, 29], "grp_indic": [21, 29], "group": [21, 29, 33, 36], "stack": [21, 29], "contigu": [21, 29], "grp1_indic": [21, 29], "grp2_indic": [21, 29], "n_group": [21, 29], "pointer": [21, 29], "consecut": [21, 29], "element": [21, 29], "delimit": [21, 29], "box": 23, "constraint": 23, "ind_": 23, "l_": [24, 27], "quasi": [24, 27], "aka": 26, "g": [29, 31, 36], "w_g": 29, "_2": 29, "resolut": 31, "silent": [31, 32, 33, 35], "refer": [31, 35], "q": [31, 32, 36], "p": [31, 36], "gidel": [31, 36], "beyond": [31, 36], "better": [31, 36], "2022": [31, 36], "http": [31, 35, 36], "arxiv": [31, 35, 36], "org": [31, 35, 36], "2204": [31, 36], "07826": [31, 36], "aistat": 31, "2021": 31, "proceed": [31, 35], "mlr": [31, 35], "press": [31, 35], "v130": 31, "bertrand21a": 31, "html": [31, 35], "github": [31, 35, 36], "mathurinm": [31, 35], "use_acc": [32, 34], "greedy_cd": 32, "keep": 32, "up": 32, "date": 32, "gram": 32, "updat": 32, "come": 32, "overhead": 32, "suit": 32, "minim": 32, "rewritten": 32, "w_init": [32, 33], "instead": [32, 33], "past": 32, "greedi": 32, "strategi": 32, "cyclic": 32, "converg": [32, 33, 35], "xw_init": 33, "minimum": [33, 35], "includ": [33, 35], "06": 34, "multi": 34, "max_pn_it": 35, "vaiter": 35, "gramfort": 35, "salmon": 35, "jmlr": 35, "2020": 35, "1907": 35, "05830": 35, "johnson": 35, "guestrin": 35, "blitz": 35, "principl": 35, "meta": 35, "icml": 35, "2015": 35, "v37": 35, "johnson15": 35, "tbjohn": 35, "blitzl1": 35, "librari": 36, "provid": 36, "Its": 36, "main": 36, "speed": 36, "million": 36, "reli": 36, "effici": 36, "flexibl": 36, "virtual": 36, "line": 36, "drop": 36, "scope": 36, "miss": 36, "etc": 36, "pleas": 36, "onlin": 36, "journal": 36, "preprint": 36, "url": 36, "pdf": 36, "year": 36, "clone": 36, "repositori": 36, "avail": 36, "contrib": 36, "git": 36, "Then": 36, "packag": 36, "pip": 36, "check": 36, "everyth": 36, "fine": 36, "do": 36, "give": 36, "error": 36, "messag": 36, "document": 36}, "objects": {"skglm": [[10, 0, 1, "", "ElasticNet"], [11, 0, 1, "", "GeneralizedLinearEstimator"], [12, 0, 1, "", "Lasso"], [13, 0, 1, "", "LinearSVC"], [14, 0, 1, "", "MCPRegression"], [15, 0, 1, "", "MultiTaskLasso"], [16, 0, 1, "", "SparseLogisticRegression"], [17, 0, 1, "", "WeightedLasso"]], "skglm.ElasticNet": [[10, 1, 1, "", "__init__"]], "skglm.GeneralizedLinearEstimator": [[11, 1, 1, "", "__init__"]], "skglm.Lasso": [[12, 1, 1, "", "__init__"]], "skglm.LinearSVC": [[13, 1, 1, "", "__init__"]], "skglm.MCPRegression": [[14, 1, 1, "", "__init__"]], "skglm.MultiTaskLasso": [[15, 1, 1, "", "__init__"]], "skglm.SparseLogisticRegression": [[16, 1, 1, "", "__init__"]], "skglm.WeightedLasso": [[17, 1, 1, "", "__init__"]], "skglm.datafits": [[18, 0, 1, "", "Huber"], [19, 0, 1, "", "Logistic"], [20, 0, 1, "", "Quadratic"], [21, 0, 1, "", "QuadraticGroup"], [22, 0, 1, "", "QuadraticSVC"]], "skglm.datafits.Huber": [[18, 1, 1, "", "__init__"]], "skglm.datafits.Logistic": [[19, 1, 1, "", "__init__"]], "skglm.datafits.Quadratic": [[20, 1, 1, "", "__init__"]], "skglm.datafits.QuadraticGroup": [[21, 1, 1, "", "__init__"]], "skglm.datafits.QuadraticSVC": [[22, 1, 1, "", "__init__"]], "skglm.penalties": [[23, 0, 1, "", "IndicatorBox"], [24, 0, 1, "", "L0_5"], [25, 0, 1, "", "L1"], [26, 0, 1, "", "L1_plus_L2"], [27, 0, 1, "", "L2_3"], [28, 0, 1, "", "MCPenalty"], [29, 0, 1, "", "WeightedGroupL2"], [30, 0, 1, "", "WeightedL1"]], "skglm.penalties.IndicatorBox": [[23, 1, 1, "", "__init__"]], "skglm.penalties.L0_5": [[24, 1, 1, "", "__init__"]], "skglm.penalties.L1": [[25, 1, 1, "", "__init__"]], "skglm.penalties.L1_plus_L2": [[26, 1, 1, "", "__init__"]], "skglm.penalties.L2_3": [[27, 1, 1, "", "__init__"]], "skglm.penalties.MCPenalty": [[28, 1, 1, "", "__init__"]], "skglm.penalties.WeightedGroupL2": [[29, 1, 1, "", "__init__"]], "skglm.penalties.WeightedL1": [[30, 1, 1, "", "__init__"]], "skglm.solvers": [[31, 0, 1, "", "AndersonCD"], [32, 0, 1, "", "GramCD"], [33, 0, 1, "", "GroupBCD"], [34, 0, 1, "", "MultiTaskBCD"], [35, 0, 1, "", "ProxNewton"]], "skglm.solvers.AndersonCD": [[31, 1, 1, "", "__init__"]], "skglm.solvers.GramCD": [[32, 1, 1, "", "__init__"]], "skglm.solvers.GroupBCD": [[33, 1, 1, "", "__init__"]], "skglm.solvers.MultiTaskBCD": [[34, 1, 1, "", "__init__"]], "skglm.solvers.ProxNewton": [[35, 1, 1, "", "__init__"]]}, "objtypes": {"0": "py:class", "1": "py:method"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "method", "Python method"]}, "titleterms": {"how": 0, "add": 0, "custom": 0, "penalti": [0, 1, 5, 6, 7, 23, 24, 25, 26, 27, 28, 29, 30], "datafit": [0, 1, 18, 19, 20, 21, 22], "api": [1, 36], "document": 1, "estim": 1, "solver": [1, 31, 32, 33, 34, 35], "exampl": 2, "galleri": 2, "time": [3, 9], "comparison": [3, 4], "scikit": 3, "learn": 3, "lasso": [3, 4, 8, 12], "weight": [4, 8], "logist": [5, 19], "regress": 5, "elast": 5, "net": 5, "minimax": 5, "concav": 5, "valu": 6, "proxim": 6, "oper": 6, "spars": 7, "recoveri": 7, "non": 7, "convex": 7, "some": 8, "zero": 8, "comput": 9, "skglm": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36], "elasticnet": 10, "generalizedlinearestim": 11, "linearsvc": 13, "mcpregress": 14, "multitasklasso": 15, "sparselogisticregress": 16, "weightedlasso": 17, "huber": 18, "quadrat": 20, "quadraticgroup": 21, "quadraticsvc": 22, "indicatorbox": 23, "l0_5": 24, "l1": 25, "l1_plus_l2": 26, "l2_3": 27, "mcpenalti": 28, "weightedgroupl2": 29, "weightedl1": 30, "andersoncd": 31, "gramcd": 32, "groupbcd": 33, "multitaskbcd": 34, "proxnewton": 35, "cite": 36, "instal": 36, "develop": 36, "version": 36}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx": 57}, "alltitles": {"How to add a custom penalty": [[0, "how-to-add-a-custom-penalty"]], "How to add a custom datafit": [[0, "how-to-add-a-custom-datafit"]], "API Documentation": [[1, "api-documentation"]], "Estimators": [[1, "estimators"]], "Penalties": [[1, "penalties"]], "Datafits": [[1, "datafits"]], "Solvers": [[1, "solvers"]], "Examples Gallery": [[2, "examples-gallery"]], "Timing comparison with scikit-learn for Lasso": [[3, "timing-comparison-with-scikit-learn-for-lasso"]], "Comparison of Lasso and Weighted Lasso": [[4, "comparison-of-lasso-and-weighted-lasso"]], "Logistic regression with Elastic net and minimax concave penalties": [[5, "logistic-regression-with-elastic-net-and-minimax-concave-penalties"]], "Value and proximal operators of penalties": [[6, "value-and-proximal-operators-of-penalties"]], "Sparse recovery with non-convex penalties": [[7, "sparse-recovery-with-non-convex-penalties"]], "Weighted Lasso with some zero weights": [[8, "weighted-lasso-with-some-zero-weights"]], "Computation times": [[9, "computation-times"]], "skglm.ElasticNet": [[10, "skglm-elasticnet"]], "skglm.GeneralizedLinearEstimator": [[11, "skglm-generalizedlinearestimator"]], "skglm.Lasso": [[12, "skglm-lasso"]], "skglm.LinearSVC": [[13, "skglm-linearsvc"]], "skglm.MCPRegression": [[14, "skglm-mcpregression"]], "skglm.MultiTaskLasso": [[15, "skglm-multitasklasso"]], "skglm.SparseLogisticRegression": [[16, "skglm-sparselogisticregression"]], "skglm.WeightedLasso": [[17, "skglm-weightedlasso"]], "skglm.datafits.Huber": [[18, "skglm-datafits-huber"]], "skglm.datafits.Logistic": [[19, "skglm-datafits-logistic"]], "skglm.datafits.Quadratic": [[20, "skglm-datafits-quadratic"]], "skglm.datafits.QuadraticGroup": [[21, "skglm-datafits-quadraticgroup"]], "skglm.datafits.QuadraticSVC": [[22, "skglm-datafits-quadraticsvc"]], "skglm.penalties.IndicatorBox": [[23, "skglm-penalties-indicatorbox"]], "skglm.penalties.L0_5": [[24, "skglm-penalties-l0-5"]], "skglm.penalties.L1": [[25, "skglm-penalties-l1"]], "skglm.penalties.L1_plus_L2": [[26, "skglm-penalties-l1-plus-l2"]], "skglm.penalties.L2_3": [[27, "skglm-penalties-l2-3"]], "skglm.penalties.MCPenalty": [[28, "skglm-penalties-mcpenalty"]], "skglm.penalties.WeightedGroupL2": [[29, "skglm-penalties-weightedgroupl2"]], "skglm.penalties.WeightedL1": [[30, "skglm-penalties-weightedl1"]], "skglm.solvers.AndersonCD": [[31, "skglm-solvers-andersoncd"]], "skglm.solvers.GramCD": [[32, "skglm-solvers-gramcd"]], "skglm.solvers.GroupBCD": [[33, "skglm-solvers-groupbcd"]], "skglm.solvers.MultiTaskBCD": [[34, "skglm-solvers-multitaskbcd"]], "skglm.solvers.ProxNewton": [[35, "skglm-solvers-proxnewton"]], "skglm": [[36, "skglm"]], "Cite": [[36, "cite"]], "Installing the development version": [[36, "installing-the-development-version"]], "API": [[36, "api"]]}, "indexentries": {"elasticnet (class in skglm)": [[10, "skglm.ElasticNet"]], "__init__() (skglm.elasticnet method)": [[10, "skglm.ElasticNet.__init__"]], "generalizedlinearestimator (class in skglm)": [[11, "skglm.GeneralizedLinearEstimator"]], "__init__() (skglm.generalizedlinearestimator method)": [[11, "skglm.GeneralizedLinearEstimator.__init__"]], "lasso (class in skglm)": [[12, "skglm.Lasso"]], "__init__() (skglm.lasso method)": [[12, "skglm.Lasso.__init__"]], "linearsvc (class in skglm)": [[13, "skglm.LinearSVC"]], "__init__() (skglm.linearsvc method)": [[13, "skglm.LinearSVC.__init__"]], "mcpregression (class in skglm)": [[14, "skglm.MCPRegression"]], "__init__() (skglm.mcpregression method)": [[14, "skglm.MCPRegression.__init__"]], "multitasklasso (class in skglm)": [[15, "skglm.MultiTaskLasso"]], "__init__() (skglm.multitasklasso method)": [[15, "skglm.MultiTaskLasso.__init__"]], "sparselogisticregression (class in skglm)": [[16, "skglm.SparseLogisticRegression"]], "__init__() (skglm.sparselogisticregression method)": [[16, "skglm.SparseLogisticRegression.__init__"]], "weightedlasso (class in skglm)": [[17, "skglm.WeightedLasso"]], "__init__() (skglm.weightedlasso method)": [[17, "skglm.WeightedLasso.__init__"]], "huber (class in skglm.datafits)": [[18, "skglm.datafits.Huber"]], "__init__() (skglm.datafits.huber method)": [[18, "skglm.datafits.Huber.__init__"]], "logistic (class in skglm.datafits)": [[19, "skglm.datafits.Logistic"]], "__init__() (skglm.datafits.logistic method)": [[19, "skglm.datafits.Logistic.__init__"]], "quadratic (class in skglm.datafits)": [[20, "skglm.datafits.Quadratic"]], "__init__() (skglm.datafits.quadratic method)": [[20, "skglm.datafits.Quadratic.__init__"]], "quadraticgroup (class in skglm.datafits)": [[21, "skglm.datafits.QuadraticGroup"]], "__init__() (skglm.datafits.quadraticgroup method)": [[21, "skglm.datafits.QuadraticGroup.__init__"]], "quadraticsvc (class in skglm.datafits)": [[22, "skglm.datafits.QuadraticSVC"]], "__init__() (skglm.datafits.quadraticsvc method)": [[22, "skglm.datafits.QuadraticSVC.__init__"]], "indicatorbox (class in skglm.penalties)": [[23, "skglm.penalties.IndicatorBox"]], "__init__() (skglm.penalties.indicatorbox method)": [[23, "skglm.penalties.IndicatorBox.__init__"]], "l0_5 (class in skglm.penalties)": [[24, "skglm.penalties.L0_5"]], "__init__() (skglm.penalties.l0_5 method)": [[24, "skglm.penalties.L0_5.__init__"]], "l1 (class in skglm.penalties)": [[25, "skglm.penalties.L1"]], "__init__() (skglm.penalties.l1 method)": [[25, "skglm.penalties.L1.__init__"]], "l1_plus_l2 (class in skglm.penalties)": [[26, "skglm.penalties.L1_plus_L2"]], "__init__() (skglm.penalties.l1_plus_l2 method)": [[26, "skglm.penalties.L1_plus_L2.__init__"]], "l2_3 (class in skglm.penalties)": [[27, "skglm.penalties.L2_3"]], "__init__() (skglm.penalties.l2_3 method)": [[27, "skglm.penalties.L2_3.__init__"]], "mcpenalty (class in skglm.penalties)": [[28, "skglm.penalties.MCPenalty"]], "__init__() (skglm.penalties.mcpenalty method)": [[28, "skglm.penalties.MCPenalty.__init__"]], "weightedgroupl2 (class in skglm.penalties)": [[29, "skglm.penalties.WeightedGroupL2"]], "__init__() (skglm.penalties.weightedgroupl2 method)": [[29, "skglm.penalties.WeightedGroupL2.__init__"]], "weightedl1 (class in skglm.penalties)": [[30, "skglm.penalties.WeightedL1"]], "__init__() (skglm.penalties.weightedl1 method)": [[30, "skglm.penalties.WeightedL1.__init__"]], "andersoncd (class in skglm.solvers)": [[31, "skglm.solvers.AndersonCD"]], "__init__() (skglm.solvers.andersoncd method)": [[31, "skglm.solvers.AndersonCD.__init__"]], "gramcd (class in skglm.solvers)": [[32, "skglm.solvers.GramCD"]], "__init__() (skglm.solvers.gramcd method)": [[32, "skglm.solvers.GramCD.__init__"]], "groupbcd (class in skglm.solvers)": [[33, "skglm.solvers.GroupBCD"]], "__init__() (skglm.solvers.groupbcd method)": [[33, "skglm.solvers.GroupBCD.__init__"]], "multitaskbcd (class in skglm.solvers)": [[34, "skglm.solvers.MultiTaskBCD"]], "__init__() (skglm.solvers.multitaskbcd method)": [[34, "skglm.solvers.MultiTaskBCD.__init__"]], "proxnewton (class in skglm.solvers)": [[35, "skglm.solvers.ProxNewton"]], "__init__() (skglm.solvers.proxnewton method)": [[35, "skglm.solvers.ProxNewton.__init__"]]}})