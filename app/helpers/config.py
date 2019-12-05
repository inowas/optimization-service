from pathlib import PurePosixPath

# Database to our optimization postgres
DATABASE_URL = "postgresql+psycopg2://root:root@postgres:5432/optimization"

# JSON_SCHEMA_UPLOAD = "./json_schema/schema_upload.json"
# JSON_SCHEMA_MODFLOW_OPTIMIZATION = "./json_schema/schema_modflow_optimization.json"
HTTPS_STRING = "https://"
# Schema that has to be validated against the uploaded json
SCHEMA_INOWAS_OPTIMIZATION = PurePosixPath("schema.inowas.com/optimization/optimization_project.json")
# Schema explicitly for the modflowmodeldata
SCHEMA_MODFLOW_MODEL_DATA = PurePosixPath("schema.inowas.com/modflow/packages/modflow_model_data.json")

# Folder for optimization data
OPTIMIZATION_DATA = "/optimization-data/"
# optimization folder
OPTIMIZATION_FOLDER = "optimization"
# calculation folder
CALCULATION_FOLDER = "calculation"

# folder for parameters
INDIVIDUAL_PARAMETERS_FOLDER = "individual_parameters"

# Optimization data name
ODATA_FILENAME = "optimization_parameters"

# Model data name
MDATA_FILENAME = "model_data"

# Calculation input ext
CALC_INPUT_EXT = "_input"
# Calculation output ext
CALC_OUTPUT_EXT = "_output"

# File ending for json
JSON_ENDING = ".json"

# Strings for optimization states
# State when task is added to optimization table
OPTIMIZATION_START = "optimization_start"
# State when task is taken by the manager and processed
OPTIMIZATION_RUN = "optimization_run"
# State when optimization is finished
OPTIMIZATION_FINISH = "optimization_finish"
# State when error appeared
OPTIMIZATION_ABORT = "optimization_abort"
# State when optimization should stop
OPTIMIZATION_STOP = "optimization_stop"

# Strings for calculation states
# State when task is added to calculation table
CALCULATION_START = "calculation_start"
# State when task is taken by the manager and processed
CALCULATION_RUN = "calculation_run"
# State when optimization is finished
CALCULATION_FINISH = "calculation_finish"
# State when error appeared
CALCULATION_ABORT = "calculation_abort"

# Initial fitness (should be really bad)
INITIAL_SCALAR_FITNESS = 999_999.0

OPTIMIZATION_TYPE_EVOLUTION = "GA"
OPTIMIZATION_TYPE_LINEAR = "Simplex"

MAX_STORING_TIME_OPTIMIZATION_TASKS = 20  # in days

MISSING_DATA_VALUE = -9999

NUMBER_OF_SOLUTIONS = 10  # could be used for fixed number of solutions

STATUS_REGULAR_CALCULATION = 0  # code for regular calculation
STATUS_ERROR_CALCULATION = 400  # code for flopy.run_model returning False
