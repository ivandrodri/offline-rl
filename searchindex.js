Search.setIndex({"alltitles": {"Addressing out of distribution issues": [[4, "addressing-out-of-distribution-issues"]], "Appendix: Short review of some popular offline RL algorithms": [[4, "appendix-short-review-of-some-popular-offline-rl-algorithms"]], "Batch Constrained deep Q-learning (BCQ) algorithm": [[4, "batch-constrained-deep-q-learning-bcq-algorithm"]], "Conservative Q-Learning (CQL) algorithm": [[4, "conservative-q-learning-cql-algorithm"]], "Data analysis": [[5, "data-analysis"]], "Exercise I": [[5, "exercise-i"]], "Exercise II": [[5, "exercise-ii"]], "Exercise: Minari data collection": [[3, null]], "Exercise: Offline RL algorithms": [[5, null]], "Final remarks": [[5, "final-remarks"]], "I - Policy constraint methods": [[4, "i-policy-constraint-methods"]], "II - Policy Regularization methods": [[4, "ii-policy-regularization-methods"]], "Implicit Q-Learning (IQL) algorithm": [[4, "implicit-q-learning-iql-algorithm"]], "Important Note": [[1, "important-note"]], "Introduction": [[4, "introduction"]], "Introduction to Offline Reinforcement Learning": [[1, null]], "MINARI Dataset": [[2, "minari-dataset"]], "Main issues": [[4, "main-issues"]], "Minari dataset structure": [[2, "minari-dataset-structure"]], "Offline RL Notes": [[0, null]], "Offline RL theory": [[4, null]], "Offline RL vs supervised learning": [[1, "offline-rl-vs-supervised-learning"]], "Offline RL/IL pipeline": [[1, "offline-rl-il-pipeline"]], "Online vs offline learning comparison.": [[1, "online-vs-offline-learning-comparison"]], "Open Source Datasets libraries for offline RL": [[2, null]], "Open X-Embodiment Repository": [[2, "open-x-embodiment-repository"]], "Overview": [[4, "overview"]], "Problem Overview: Offline vs. Online RL": [[1, "problem-overview-offline-vs-online-rl"]], "Q-Transformer": [[4, "q-transformer"]], "RL Unplugged dataset": [[2, "rl-unplugged-dataset"]], "References": [[1, "references"], [2, "references"], [3, "references"], [4, "references"]], "STEP 1: Create the environment": [[5, "step-1-create-the-environment"], [5, "id1"]], "STEP 1: Create the environments": [[3, "step-1-create-the-environments"]], "STEP 2: Create Minari datasets": [[3, "step-2-create-minari-datasets"], [5, "step-2-create-minari-datasets"], [5, "id2"]], "STEP 3: Feed data into replay buffer": [[5, "step-3-feed-data-into-replay-buffer"], [5, "id3"]], "STEP 3: Feed dataset to Tianshou ReplayBuffer": [[3, "step-3-feed-dataset-to-tianshou-replaybuffer"]], "STEP 4-5: Select offline policies and training": [[5, "step-4-5-select-offline-policies-and-training"]], "STEP 4: Select offline policies and training": [[5, "step-4-select-offline-policies-and-training"]], "Summary": [[4, "summary"]], "Summary and conclusions": [[5, "summary-and-conclusions"]], "Summary:": [[4, "id1"]], "Useful minari methods": [[2, "useful-minari-methods"]], "a) Non-implicit or Direct": [[4, "a-non-implicit-or-direct"]], "b) Implicit": [[4, "b-implicit"]]}, "docnames": ["intro", "nb_0_IntroOfflineRL", "nb_1_offline_RL_dataset_libraries", "nb_2_data_dollection", "nb_3_offline_RL_theory", "nb_4_Offline_rl_exercises"], "envversion": {"sphinx": 62, "sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1}, "filenames": ["intro.md", "nb_0_IntroOfflineRL.ipynb", "nb_1_offline_RL_dataset_libraries.ipynb", "nb_2_data_dollection.ipynb", "nb_3_offline_RL_theory.ipynb", "nb_4_Offline_rl_exercises.ipynb"], "indexentries": {}, "objects": {}, "objnames": {}, "objtypes": {}, "terms": {"": [1, 2, 3, 4, 5], "0": [1, 2, 3, 4, 5], "000000": 5, "01440554": 3, "022067": 5, "090235": 5, "1": [1, 2, 4], "10": 3, "100": [3, 5], "1000": 5, "12": [1, 2, 3, 4, 5], "125": 5, "128": 5, "2": [1, 2, 4], "20": 5, "2000": 5, "2017": 4, "2019": 4, "2020": 4, "2021": [1, 2, 4], "2023": [1, 2], "207813": 5, "22": 2, "250": 5, "2a": 4, "2d": [1, 3, 5], "3": [1, 2, 4], "33": 2, "375": 5, "4": [1, 2, 4], "40": 5, "43": 5, "5": [1, 2, 4], "500": 3, "6": [2, 4, 5], "60": 5, "600": 5, "64": 3, "7": [4, 5], "8x8": 3, "A": [1, 2, 3, 4], "And": 2, "As": [1, 2, 4], "Be": 1, "But": [1, 4], "For": [1, 2, 4], "If": [0, 1, 4], "In": [1, 2, 3, 4, 5], "It": [1, 2, 3, 4, 5], "Its": 2, "On": [1, 2, 4], "One": [1, 4, 5], "The": [1, 2, 3, 4, 5], "There": 1, "These": [0, 1, 2, 4], "To": [3, 4], "With": [1, 4], "_": [1, 3, 4, 5], "_0": 4, "_1": 4, "_2": 4, "_4": 4, "__name__": 3, "_collected_data_nb_92": 3, "_expert": 5, "_k": 4, "_longer_path": 5, "_q": 4, "_short_path": 5, "_stiching_property_i": 5, "_suboptim": 5, "a_": 4, "a_0": [1, 4], "a_1": 1, "a_4": 4, "a_i": 4, "a_t": [1, 4], "abl": [1, 2, 4, 5], "about": [1, 4, 5], "abov": 1, "absorb": 4, "abund": 5, "academ": 2, "acceler": [1, 4], "accept": 1, "access": [1, 2, 3, 4, 5], "account": 4, "accur": [1, 2, 4, 5], "achiev": [2, 4, 5], "across": [1, 5], "act": [3, 4], "action": [1, 3, 4, 5], "action_i": 2, "activ": 1, "actor": 4, "ad": 1, "adapt": [1, 4], "addition": 1, "address": [1, 5], "adjust": 4, "adopt": 1, "adroithandhamm": 3, "advanc": 2, "advantag": [1, 4], "affect": 4, "aforement": 4, "after": 1, "again": [4, 5], "agent": [1, 2, 3, 4, 5], "agreement": 4, "aid": 5, "aim": [1, 2, 4], "al": [1, 2, 3, 4], "alert": 4, "algorithm": [1, 2], "align": 4, "all": [0, 1, 4], "allow": [1, 2, 4], "almost": 1, "along": 1, "alpha": 4, "alreadi": [1, 4, 5], "also": [1, 2, 3, 4], "altern": 4, "although": 2, "alwai": 1, "amount": [1, 5], "ampl": 5, "an": [1, 2, 3, 4, 5], "anal": 4, "analyz": [1, 3], "ani": [1, 2, 3, 4], "anoth": [4, 5], "anywai": 1, "apart": 1, "api": [2, 3], "appear": [2, 4], "appli": [1, 4, 5], "applic": [1, 4, 5], "approach": [1, 2, 4, 5], "approxim": 4, "ar": [1, 2, 3, 4, 5], "architectur": 4, "area": [1, 2], "arg": 4, "argmin": 4, "argmin_": 1, "aris": [2, 4], "around": [0, 1, 3, 4], "arrai": 3, "aspect": 4, "assess": 5, "assum": [1, 4], "attach": 1, "attain": 4, "attempt": [1, 4], "attent": 2, "attract": 2, "author": 4, "autoencod": 4, "automot": 2, "autonom": [1, 2, 5], "autoreload": [1, 2, 3, 4, 5], "avail": 5, "available_obstacl": 5, "avoid": [1, 4], "awac": 4, "awr": 4, "b": 5, "back": 1, "backup": 4, "bad": 4, "balanc": 4, "base": [1, 2, 4], "batch": [1, 3], "batch_siz": 5, "bc": [1, 4, 5], "bcq": 5, "bcq_discret": 5, "becaus": [1, 2, 5], "becom": [2, 4, 5], "befor": [1, 2, 4, 5], "beforehand": 1, "begin": 4, "behavior": [1, 2, 3, 4, 5], "behavior_8x8_eps_greedy_4_0_to_7_7": 5, "behavior_polici": [3, 5], "behavior_policy_grid_world": 3, "behavior_policy_i": 5, "behavior_policy_ii": 5, "behavior_policy_registri": [3, 5], "behaviorpolicy2dgridfactori": [3, 5], "behaviorpolicytyp": 5, "being": [1, 4], "bellman": 4, "belong": 1, "below": [1, 4], "benchmark": 2, "benefici": 1, "berkelei": 2, "best": [1, 2], "best_reward": 5, "beta": [1, 4], "better": [2, 4], "between": [1, 5], "beyond": 1, "bf": 4, "bia": 4, "bias": 2, "big": [1, 4], "bit": 4, "blog": [2, 4], "bool": 5, "both": [4, 5], "bound": 4, "brain": 2, "break": 3, "bring": 4, "broader": 4, "browser": [3, 5], "buffer": 4, "buffer_data": [3, 5], "build": 1, "c": [1, 2, 4, 5], "call": [1, 2], "camera": 2, "can": [1, 2, 3, 4, 5], "candid": 4, "cannot": [1, 4, 5], "capabl": [1, 5], "captur": [1, 2, 3, 4, 5], "car": [1, 2], "care": 1, "carri": 1, "carrot": 1, "case": [1, 4, 5], "catastroph": 4, "categori": 4, "caus": 4, "cdot": 4, "centric": 5, "certain": 4, "challeng": [1, 2, 4, 5], "chanc": 4, "chang": 5, "character": 4, "cheaper": 1, "check": 5, "cheetah": 4, "children_dataset_nam": 5, "choic": [2, 4, 5], "chosen": 5, "ci": 5, "class": [2, 5], "classic": 2, "classifi": 4, "clear": 2, "clearer": 1, "clever": 4, "clip": 4, "clone": [1, 4], "close": [1, 2, 4, 5], "code": [1, 3, 4], "collect": [1, 2, 4, 5], "collis": 1, "color": [1, 4], "combin": [1, 2, 4, 5], "combine_dataset": 2, "combined_data_sets_offline_rl": 5, "combined_dataset": 5, "combined_dataset_identifi": 5, "come": [1, 2, 4, 5], "comment": 4, "common": [2, 4], "commun": [1, 2], "compar": [1, 5], "complet": 4, "complex": [1, 2, 4, 5], "complic": 1, "compress": 2, "comput": [1, 3, 4, 5], "computation": 4, "con": 4, "concept": [1, 4], "concern": 1, "config_combined_data": 5, "configur": 5, "congratul": 3, "connect": [4, 5], "consequ": 5, "consid": [1, 4], "consider": 4, "consist": 4, "constraint": 1, "construct": 4, "consum": 1, "consumpt": 1, "contact": 1, "contain": [0, 1, 2], "continu": 1, "contrast": [1, 4], "control": [1, 2, 4], "convert": 2, "coordin": 3, "copi": 4, "core": [1, 4], "correl": 4, "cost": 1, "costli": [1, 2], "could": [1, 2, 4], "cover": [4, 5], "cpu": 5, "cql": 5, "cql_discret": 5, "craft": [2, 5], "creat": 1, "create_combined_minari_dataset": 5, "create_dataset": 2, "create_dataset_from_buff": 2, "create_minari_dataset": 3, "criteria": 2, "critic": [1, 4, 5], "crucial": [1, 2, 4, 5], "cucumb": 1, "cumul": 4, "current": 1, "custom": [1, 2], "custom2dgridenv": 3, "custom_2d_grid_env": [3, 5], "custom_env": [3, 5], "custom_envs_registr": [3, 5], "cycl": 1, "d": [1, 4, 5], "d4rl": [2, 3], "d_": [1, 4], "danger": 1, "data": [1, 2, 4], "data_collected_data_nb_92": 3, "data_s": 5, "data_set_config": 3, "data_set_identifier_i": [3, 5], "data_set_identifier_ii": 5, "data_set_nam": [3, 5], "datacollector": [2, 3, 5], "datacollectorv0": [3, 5], "dataset": [1, 4], "dataset_avail": 5, "dataset_identifi": [3, 5], "ddq": 4, "deal": [1, 4, 5], "decis": 1, "deep": [2, 3], "deeper": 4, "defin": [1, 2, 4], "definit": 1, "demand": 5, "demonstr": [1, 2], "denot": 1, "densiti": 4, "deprec": [3, 5], "deprecationwarn": [3, 5], "deriv": [1, 4], "describ": 0, "descript": 3, "design": [1, 2], "destroi": 5, "detail": 4, "detect": 1, "determin": 1, "determinist": 2, "deterministic_8x8": [3, 5], "develop": 1, "deviat": 4, "devic": 5, "diagnosi": 1, "did": [1, 2, 5], "didn": 2, "differ": [1, 2, 3, 4], "difficult": [1, 2, 4], "dimens": 3, "dimension": [1, 3, 4], "dimenst": 4, "direct": 3, "directli": [1, 4], "directori": 2, "discount": [1, 4], "discourag": 4, "discov": [1, 4], "discoveri": 4, "discrep": 1, "discuss": [2, 4], "displai": 3, "distinct": 1, "distribut": [1, 2, 3, 5], "diverg": 4, "divers": [2, 5], "dk_1": 4, "dkl": 4, "dkl_2": 4, "dnn": [1, 4], "do": [1, 5], "docker": 0, "document": [2, 3], "doe": [3, 5], "doesn": [1, 4], "domain": [1, 2], "don": [1, 3, 4], "done": [3, 5], "doubl": 4, "down": 3, "download": [3, 5], "drift": 5, "drive": [1, 2, 4, 5], "driven": [1, 2, 3], "due": [1, 5], "dure": [1, 4], "dynam": [4, 5], "e": [1, 2, 4, 5], "e_": 4, "each": [3, 4], "earlier": 4, "easi": [2, 4], "easier": [1, 2, 4], "easili": 4, "edg": 5, "edit": 0, "effect": [1, 2, 4, 5], "effici": [1, 2, 4, 5], "egl": 3, "elem": 3, "elif": 5, "els": [3, 5], "embodi": 5, "emploi": 4, "enabl": 4, "encod": 3, "encount": 1, "encourag": 4, "end": [2, 4], "energi": 1, "enhanc": [4, 5], "enough": 4, "ensur": 4, "entir": [1, 4], "env": [3, 5], "env_2d_grid_initial_config": 5, "env_2d_grid_initial_config_i": 5, "env_2d_grid_initial_config_ii": 5, "env_list": 3, "env_nam": [3, 5], "env_or_env_nam": [3, 5], "env_wrapp": 5, "envfactori": [3, 5], "environ": [1, 2, 4], "environment": 1, "episod": [1, 3, 5], "epoch": 5, "epsilon": [1, 4], "eq": 4, "equal": 2, "equat": [4, 5], "error": [1, 5], "especi": [1, 2, 4, 5], "essenti": 1, "estim": [1, 4], "et": [1, 2, 3, 4], "eta": 4, "evalu": [1, 2, 4, 5], "even": [1, 4], "everyth": 2, "everywher": 2, "evid": 5, "evolv": [4, 5], "examin": 5, "exampl": [1, 3, 4], "except": 2, "exclud": [1, 4], "execut": 0, "exemplifi": 5, "exercis": [1, 4], "exhaust": 4, "exhibit": 4, "exist": [1, 2], "exp": 4, "expect": [1, 4], "expectil": 4, "expens": [1, 4], "experi": 1, "expert": [1, 2, 4, 5], "explain": 2, "explan": 1, "explicit": 4, "explor": [1, 2, 3, 4, 5], "extens": 5, "extract": 2, "f": 3, "facilit": [1, 4], "fair": 1, "fall": 4, "fals": [3, 5], "familiar": [2, 3], "far": [1, 4], "fast": 1, "faster": 1, "feasibl": 1, "feed": 1, "feedback": [1, 5], "feel": 0, "few": 4, "field": [1, 4], "fig": [1, 4], "figur": [1, 2, 4], "file": 2, "fill": 4, "filterwarn": [3, 5], "final": 4, "final_st": 5, "final_state_polici": 5, "financ": 1, "financi": 1, "find": [1, 2, 4, 5], "first": [3, 4, 5], "fix": [1, 2, 4], "focu": [2, 4], "focus": [1, 4, 5], "folder": [1, 2], "follow": [1, 2, 3, 4], "forecast": 5, "form": 1, "format": [1, 2], "forward": 1, "four": 5, "fp": 3, "frac": 4, "framework": 4, "free": 0, "from": [1, 2, 3, 4, 5], "fu": [2, 3], "fulli": 1, "function": [1, 2, 4], "further": [1, 4], "furthermor": 5, "futil": 4, "g": 2, "g_": 4, "gain": [2, 5], "gamma": [1, 4], "gap": 4, "gather": [1, 2], "gcp": 2, "gener": [1, 2, 4], "generate_custom_minari_dataset": [3, 5], "generate_minari_dataset": [3, 5], "get": [1, 2, 3, 4, 5], "get_env": [3, 5], "get_state_action_data_and_policy_grid_distribut": [3, 5], "get_trained_policy_path": 5, "getenv": 5, "github": 2, "give": [1, 2, 3, 4], "given": [1, 4], "go": [1, 2, 4, 5], "goal": [1, 2, 4, 5], "goe": [1, 5], "good": [1, 3, 4], "googl": 2, "grasp": 5, "greedili": 4, "green": [1, 2], "grid": [1, 3, 5], "grid2dinitialconfig": 5, "grid_2d_8x8_discret": [3, 5], "grid_config": 5, "grid_world_obstacl": 3, "gridworld": [3, 5], "groot": 2, "group": 2, "guarante": 4, "guid": 4, "gulcehr": 2, "gym": 3, "gymnasium": [2, 3], "h": [1, 2], "h5py": 2, "ha": [2, 4], "half": 4, "halfcheetah": 3, "hand": [1, 2, 5], "handl": 2, "happen": 4, "hard": 1, "hat": 4, "have": [1, 2, 3, 4, 5], "hbox": 4, "hdf5": 2, "health": 1, "healthcar": [1, 5], "help": [1, 2, 4], "henc": 4, "here": [2, 4], "hesit": 1, "higer": 5, "high": [1, 4], "higher": [4, 5], "highest": 1, "highli": [1, 4], "histor": [1, 2], "homework": 3, "horizon": 1, "horizont": 3, "host": 3, "hot": 3, "how": [1, 2, 4, 5], "howev": [1, 4, 5], "human": [1, 2], "humanoid": [2, 3], "hyperparamet": 5, "i": [1, 2, 3], "id": 3, "idea": [1, 3, 4], "ideal": 1, "identifi": 2, "identifier_combined_dataset": 5, "ignor": [3, 5], "iii": 4, "illustr": [1, 4], "imit": [1, 2, 4, 5], "imitation_learn": 5, "imitation_policy_sampl": 5, "impact": 1, "implement": 4, "impli": 4, "import": [3, 4, 5], "importantli": 5, "impos": 1, "imposs": [1, 5], "improv": [1, 2, 4], "inaccur": 5, "includ": [2, 4, 5], "incorpor": 4, "increas": 1, "inde": 4, "indic": 3, "industri": 1, "infer": 4, "info": 3, "info_i": 2, "inform": 3, "infrequ": 4, "infti": [1, 4], "initial_st": 5, "initial_state_policy_i": 5, "initial_state_policy_ii": 5, "inlin": [1, 2, 3, 4, 5], "instal": 0, "instanc": 2, "instead": [1, 3, 5], "integ": 3, "integr": 4, "intellig": 5, "intend": 2, "interact": [1, 2], "interest": 4, "internet": 2, "interpret": 4, "intervent": 1, "intric": 1, "introduc": [1, 2, 4], "intuit": [1, 4], "invertedpendulum": 3, "invest": 1, "involv": [1, 2, 4, 5], "ipynb": 1, "is_not_ci": 5, "isinst": 3, "isn": 1, "issu": [1, 2, 5], "iter": 4, "its": 5, "j": [1, 4], "jump": 4, "jupyt": 0, "just": [1, 2, 4], "justin": [2, 3], "k": 4, "kei": [1, 2, 4], "kind": 4, "kl": 4, "know": 4, "knowledg": [1, 4], "known": 4, "kullback": 4, "kumar": 4, "l": 4, "l_": [1, 4], "l_1": 4, "l_2": 4, "l_q": 4, "l_v": 4, "lab": 2, "label": 4, "lack": 1, "lagrang": 4, "lambda": 4, "landscap": 5, "larg": [1, 2, 4, 5], "larger": 5, "later": [1, 2, 4], "ldot": [1, 4], "le": 1, "lead": [1, 2, 4], "learn": [2, 3, 5], "left": [1, 3, 4, 5], "leftarrow": 4, "leibler": 4, "len": [3, 5], "len_buff": 5, "leq": 4, "let": [1, 2, 3, 4, 5], "level": [1, 4], "leverag": 5, "levin": [1, 4], "li": 1, "librari": 1, "lidar": 1, "like": [1, 2, 4, 5], "limit": [1, 4], "line": [1, 4], "link": 2, "list": [2, 3, 4], "list_local_dataset": 2, "littl": [1, 5], "ll": [1, 2, 5], "load": [2, 5], "load_buffer_minari": [3, 5], "load_dataset": 2, "load_ext": [1, 2, 3, 4, 5], "load_state_dict": 5, "localhost": 3, "log": 4, "log_nam": 5, "log_path": 5, "logdir": 3, "longer": 4, "look": [0, 2, 3], "loop": 4, "lot": 2, "low": 4, "lower": 4, "machin": 2, "made": 2, "mai": [1, 2, 4], "main": [1, 2], "mainli": 2, "major": [1, 2, 5], "make": [1, 2, 3, 4, 5], "manag": 1, "mani": [1, 2, 4, 5], "manipul": [1, 2], "manner": 1, "manual": 1, "manufactur": 2, "map_loc": 5, "market": 1, "markov": 1, "markovian": 2, "match": 2, "math": 4, "mathbb": [1, 4], "mathcal": 4, "mathemat": 4, "matplotlib": [1, 2, 3, 4, 5], "max": 4, "max_": 4, "maxim": 4, "maximum": 4, "md": 0, "mean": [2, 4], "meaning": 4, "meat": 1, "medium": 2, "memori": 2, "mention": 1, "merg": 2, "meta": 2, "metadata": [2, 5], "method": [1, 5], "methodologi": 4, "mid": 4, "might": [1, 2], "mimic": 4, "min": 4, "min_": 4, "min_q": 4, "minari": 1, "mind": 2, "minim": [1, 4], "minimum": 4, "mirror": 1, "misinform": 4, "miss": [1, 4], "mitig": 1, "mix": 3, "mixtur": 1, "model": [1, 2, 4, 5], "modifi": 4, "more": [1, 4, 5], "moreov": 4, "most": [2, 4], "mostli": 1, "move": [1, 5], "move_left": 5, "move_right": 5, "move_up": 5, "mu": 4, "much": 4, "mujoco": 2, "mujoco_gl": 3, "multi": [2, 4], "multipl": [2, 4], "multipli": 4, "multitask": [2, 5], "must": 1, "my": 0, "n": [1, 4], "nair": 4, "name": 4, "name_expert_data": 5, "narrow": 2, "navig": [1, 5], "nb_3_offline_rl_theori": 1, "nb_95": 4, "ne": 1, "necessarili": [1, 4], "necessit": 4, "need": [1, 2, 3, 4], "network": 4, "neural": 4, "nevertheless": 5, "new": [1, 2, 4, 5], "next": 4, "nice": 4, "nois": [2, 4, 5], "non": 2, "normal": 5, "notat": 1, "note": [4, 5], "notebook": [0, 1, 3, 4], "notic": 5, "novel": 4, "now": [1, 2, 3, 5], "num_collected_point": [3, 5], "num_epoch": 5, "num_fram": [3, 5], "num_steps_i": [3, 5], "num_steps_ii": 5, "number": [1, 3, 5], "number_test_env": 5, "nvidia": 2, "o": [1, 3, 4, 5], "ob": 3, "object": 2, "obs_next": 3, "observ": [1, 2, 3, 4], "obst": 3, "obst_free_8x8": [3, 5], "obstacl": [1, 3, 5], "obstacle_8x8_top_right": 3, "obstacle_8x8_wall_with_door": 5, "obstacle_select": 5, "obstacles_2d_grid_regist": [3, 5], "obstacletyp": [3, 5], "occur": [1, 4], "octob": 2, "off": [1, 4, 5], "offer": 2, "offilin": 5, "offlin": 3, "offline_data": [3, 5], "offline_polici": [3, 5], "offline_policy_config": 5, "offline_rl": [1, 3, 5], "offline_rl_polici": 5, "offline_train": 5, "offline_training_hyperparam": 5, "offlinerltrain": 5, "offlinetraininghyperparam": 5, "offpolici": 4, "offpolicy_rend": [3, 5], "often": [1, 2, 4], "omega": 4, "onc": [1, 4, 5], "one": [1, 2, 3, 4, 5], "ones": [4, 5], "ongo": 4, "onion": 1, "onli": [1, 2, 4, 5], "onlin": [2, 4, 5], "op": 4, "open": [1, 4, 5], "oper": [1, 4], "opportun": 5, "opposit": 4, "optim": [1, 2, 4, 5], "option": [1, 4, 5], "order": [1, 5], "organ": 2, "origin": 1, "other": [1, 2, 4, 5], "otherwis": 1, "our": [1, 2, 3, 4, 5], "out": [1, 2, 5], "outperform": [1, 5], "output": 4, "outsid": [1, 4], "over": [1, 4], "overestim": [1, 4], "overli": 2, "own": 1, "padalkar": 2, "page": 0, "pair": [4, 5], "paper": 4, "parallel": 4, "paramet": 4, "paramount": 5, "partial": [1, 2], "particular": [2, 4], "particularli": [1, 4], "partner": 2, "past": 1, "path": [2, 5], "pathlib": 5, "patienc": 1, "patient": 1, "peak": 5, "peng": 4, "percept": 5, "perfect": [2, 4], "perform": [1, 2], "person": 0, "perspect": [1, 4], "perturb": 4, "pessimist": 4, "phi": 4, "pi": [1, 4], "pi_": [1, 4], "pi_b": [1, 2, 4], "pi_k": 4, "pictur": [1, 4], "piec": 4, "pipelin": [3, 5], "plai": [1, 3, 4, 5], "pleas": 1, "plu": 4, "point": [1, 2, 3, 4, 5], "polici": [1, 2, 3], "policy_best_reward": 5, "policy_config_data_class": 5, "policy_model": 5, "policy_registri": 5, "policy_select": 5, "portfolio": 1, "portion": 5, "pose": 2, "posit": 3, "possibl": [1, 4, 5], "potenti": [4, 5], "power": [4, 5], "practic": 1, "pre": 2, "predict": 1, "prefer": 1, "prepar": 1, "preprocess": 2, "present": 1, "prevent": 4, "previou": [1, 2, 4, 5], "previous": [1, 2, 4], "primari": [1, 4], "primarili": 1, "principl": 4, "print": [3, 5], "prioriti": 4, "pro": 4, "probabl": [4, 5], "problem": 4, "process": [1, 2, 4, 5], "produc": [1, 4, 5], "program": [4, 5], "project": 2, "promin": 5, "propag": [1, 4, 5], "propel": 2, "properti": [2, 4, 5], "proport": 4, "prove": 5, "provid": [1, 2, 4, 5], "prudencio": 1, "psi": 4, "pth": 5, "purpos": 2, "q_": 4, "qquad": 4, "quad": [1, 4], "qualit": 4, "qualiti": 1, "question": [1, 5], "quick": 1, "quickli": 1, "quit": [1, 4], "r": [1, 4], "r_0": 4, "r_1": 4, "r_2": 4, "rais": 3, "random": [1, 2, 3, 4], "rang": [3, 4], "rapidli": 4, "rare": 5, "rather": 2, "reach": 5, "readm": 0, "real": [1, 2, 4, 5], "realist": [2, 4, 5], "reason": [1, 4, 5], "recov": [1, 5], "red": [1, 4], "reduc": [1, 4], "refer": 0, "refin": 1, "region": [1, 4, 5], "regist": [2, 3], "regress": 4, "reinforc": [2, 3, 4, 5], "relat": 1, "remain": 4, "rememb": 3, "remov": [3, 5], "render": [0, 3], "render_mod": [3, 5], "rendermod": [3, 5], "replac": [2, 4], "replai": 4, "repres": [1, 2, 3], "represent": 2, "requir": [0, 1, 2, 5], "research": 4, "resembl": 4, "respons": 4, "restor": 5, "restore_polici": 5, "restore_train": 5, "restrict": 4, "result": [1, 2, 4, 5], "return": [1, 4, 5], "reus": 1, "review": 1, "revisit": 4, "rew": 3, "reward": [1, 2, 4, 5], "reward_i": 2, "rgb_arrai": [3, 5], "right": [1, 3, 4], "rightarrow": 4, "risk": 4, "riski": 1, "rl_policy_model": 5, "rlpolicyfactori": 5, "robot": [1, 2, 5], "robust": 5, "role": 4, "rollout": 2, "room": 1, "rt": 2, "rule": 1, "run": 4, "s_0": [1, 4], "s_1": 1, "s_t": [1, 4], "sac": 4, "safe": [1, 4], "safer": 1, "safeti": [4, 5], "same": 4, "sampl": [1, 4], "sarsa": 4, "save": 2, "saved_policy_nam": 5, "saw": 4, "scalabl": 4, "scale": [1, 2], "scenario": [2, 4, 5], "schulman": 4, "scope": 2, "search": 2, "second": 5, "section": 2, "see": [1, 4, 5], "seem": 4, "seen": [1, 2, 4], "select": [1, 3], "select_policy_to_rend": 5, "selected_data_set": 5, "selected_environ": 3, "selected_grid_world_polici": 3, "selected_obstacl": [3, 5], "selected_offline_rl_polici": 5, "self": 1, "sens": 4, "sensor": [1, 2], "seriou": 5, "serv": 0, "set": [1, 4], "set_env_vari": [3, 5], "set_goal_point": 5, "set_new_obstacle_map": [3, 5], "set_random_se": [1, 2, 3, 4, 5], "set_starting_point": 5, "setup": [2, 5], "sever": [1, 4], "shift": [1, 4], "short": 2, "should": [1, 2, 3, 4], "show_progress": 5, "shown": [1, 4], "side": 4, "signific": [2, 4, 5], "significantli": 4, "sim": [1, 4], "simeq": 1, "similar": [2, 4, 5], "simpl": 4, "simple_grid": 3, "simpler": 1, "simplest": 1, "simplifi": 1, "simul": [1, 2, 5], "simultan": 4, "sinc": [1, 2, 4], "situat": [1, 2, 4], "size": [1, 2], "skill": 1, "slice": 2, "slide": [1, 4], "slightli": 4, "small": [4, 5], "smaller": 2, "snapshot_env": [3, 5], "so": [1, 2, 4, 5], "soft": 4, "sole": [1, 5], "solut": [1, 4], "solv": [1, 2, 4], "some": [1, 5], "someth": 4, "sometim": [1, 4], "somewhat": 2, "soon": 1, "soup": 1, "sourc": 3, "space": [1, 4], "spars": 2, "spec": 3, "specif": [1, 2, 5], "split": 2, "src": [1, 3, 5], "standard": [1, 2], "start": [0, 5], "start_0_0": 5, "start_2_0": 5, "state": [1, 2, 3, 4, 5], "state_action_count_data": [3, 5], "state_i": 2, "static": 1, "step": [1, 4], "step_per_epoch": 5, "still": [2, 4], "stitch": [2, 4, 5], "stop": [1, 5], "storag": 2, "store": [2, 4], "str": 5, "straightforward": 1, "strategi": [1, 4, 5], "strict": 1, "strong": 4, "structur": 3, "struggl": 4, "studi": [2, 5], "style": 4, "suboptim": [1, 2, 4, 5], "suboptimal_8x8": 3, "subset": [1, 2], "substanti": 1, "success": 2, "suffer": 4, "suffici": 1, "suggest": 2, "suit": [2, 4], "suitabl": [2, 4], "sum": 4, "sum_": [1, 4], "sum_a": 4, "sum_t": 1, "summar": 4, "summari": 1, "superior": 1, "supervis": 2, "support": [2, 3, 4, 5], "suppos": 2, "survei": 1, "svi": 4, "sy": [3, 5], "symptom": 1, "t": [1, 2, 3, 4], "tackl": 1, "tag": [1, 3, 4, 5], "tailor": 2, "take": [3, 4, 5], "taken": 4, "targ": 4, "target": 5, "target_st": 5, "task": [1, 2, 4, 5], "tau": [1, 4], "tau_i": 1, "taxonomi": 1, "teach": 2, "technic": 4, "techniqu": [1, 4], "tend": [1, 5], "tendenc": [4, 5], "tensorboard": 3, "term": 1, "termin": [1, 3], "termination_i": 2, "test": [1, 2], "test_reward": 5, "text": [1, 4], "than": [1, 2, 4], "thei": [2, 4, 5], "them": [1, 2, 5], "theoret": 4, "therefor": 4, "theta": [1, 4], "theta_i": 4, "thi": [1, 2, 3, 4, 5], "thing": 4, "think": [1, 5], "those": [1, 4, 5], "though": 4, "three": 4, "through": [2, 4], "ti": 2, "tianshou": 1, "time": [1, 3, 4, 5], "too": 4, "top": 4, "torch": 5, "total": 1, "toward": 5, "trade": 1, "train": [1, 2, 4], "trainedpolicyconfig": 5, "training_interfac": 5, "trajectori": [1, 2, 4, 5], "transfer": [2, 5], "transit": 1, "treat": 1, "treatment": 1, "true": [1, 5], "truli": 1, "truncat": 3, "truncation_i": 2, "trust": 4, "try": [1, 4], "turn": 5, "tutori": [1, 4], "two": [1, 2, 4, 5], "type": [1, 2, 3, 4, 5], "typic": [1, 2, 4], "u": [1, 2, 4], "uc": 2, "unbalanc": 4, "uncertainti": 4, "unconvent": 4, "under": 4, "underrepres": 4, "understand": 4, "undesir": 4, "undirect": 2, "unknown": 1, "unless": 5, "unlik": 1, "unpredict": [4, 5], "unrel": 1, "unrepres": 4, "unwrap": 3, "up": 3, "updat": 4, "us": [0, 1, 3, 4, 5], "usag": 1, "user": 2, "usual": [1, 4], "util": [1, 2, 3, 4, 5], "v": 4, "v0": [3, 5], "v1": 3, "v4": 3, "v_": 4, "vae": 4, "valid": [1, 4], "valu": [3, 4, 5], "valuabl": [2, 5], "valueerror": 3, "vari": 1, "variabl": 4, "varianc": 1, "variat": 4, "variou": [1, 2, 4, 5], "vast": [2, 5], "ve": 4, "vector": 3, "vehicl": 1, "veri": [1, 5], "vert": 4, "vertic": 3, "viabl": [1, 5], "video": [2, 3, 5], "view": 4, "visual": [3, 5], "wa": 2, "wai": [2, 3, 4], "want": [0, 1, 4, 5], "warn": [3, 5], "we": [1, 2, 3, 4, 5], "weight": 4, "welcom": 0, "well": [2, 4, 5], "were": [1, 4], "what": [1, 2, 4, 5], "when": [1, 4, 5], "where": [1, 2, 4, 5], "wherea": 1, "whether": [1, 2, 4], "which": [1, 2, 4, 5], "while": [1, 2, 4], "who": 1, "why": [1, 4, 5], "wide": 1, "widespread": 1, "widget_list": [3, 5], "window": 2, "within": [1, 2, 4], "without": [1, 2, 4, 5], "won": 4, "word": 4, "work": [1, 2, 4], "workspac": [3, 5], "world": [1, 2, 3, 4, 5], "worri": 1, "worth": 4, "would": [1, 3, 4, 5], "wrapper": 2, "written": 4, "wrong": 1, "x": 5, "x_1": 3, "x_2": 3, "xi_": 4, "y": 4, "ye": 1, "yellow": 2, "yet": 4, "yield": 4, "you": [0, 1, 3, 4, 5], "your": [1, 2, 3, 4, 5], "z": 4, "zero": 4}, "titles": ["Offline RL Notes", "Introduction to Offline Reinforcement Learning", "Open Source Datasets libraries for offline RL", "Exercise: Minari data collection", "Offline RL theory", "Exercise: Offline RL algorithms"], "titleterms": {"1": [3, 5], "2": [3, 5], "3": [3, 5], "4": 5, "5": 5, "address": 4, "algorithm": [4, 5], "analysi": 5, "appendix": 4, "b": 4, "batch": 4, "bcq": 4, "buffer": 5, "collect": 3, "comparison": 1, "conclus": 5, "conserv": 4, "constrain": 4, "constraint": 4, "cql": 4, "creat": [3, 5], "data": [3, 5], "dataset": [2, 3, 5], "deep": 4, "direct": 4, "distribut": 4, "embodi": 2, "environ": [3, 5], "exercis": [3, 5], "feed": [3, 5], "final": 5, "i": [4, 5], "ii": [4, 5], "il": 1, "implicit": 4, "import": 1, "introduct": [1, 4], "iql": 4, "issu": 4, "learn": [1, 4], "librari": 2, "main": 4, "method": [2, 4], "minari": [2, 3, 5], "non": 4, "note": [0, 1], "offlin": [0, 1, 2, 4, 5], "onlin": 1, "open": 2, "out": 4, "overview": [1, 4], "pipelin": 1, "polici": [4, 5], "popular": 4, "problem": 1, "q": 4, "refer": [1, 2, 3, 4], "regular": 4, "reinforc": 1, "remark": 5, "replai": 5, "replaybuff": 3, "repositori": 2, "review": 4, "rl": [0, 1, 2, 4, 5], "select": 5, "short": 4, "some": 4, "sourc": 2, "step": [3, 5], "structur": 2, "summari": [4, 5], "supervis": 1, "theori": 4, "tianshou": 3, "train": 5, "transform": 4, "unplug": 2, "us": 2, "v": 1, "x": 2}})