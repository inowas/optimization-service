# Database to our optimization postgres
DATABASE_URL = "postgresql+psycopg2://root:root@postgres:5432/optimization"

# Schema that has to be validated against the uploaded json
JSON_SCHEMA_UPLOAD = "./json_schema/schema_upload.json"

# Folder for optimization data
OPTIMIZATION_DATA = "/optimization-data/"
CALCULATION_DATA = "/calculation-data/"

# Optimization ext
OPTIMIZATION_FILE = "optimization"
# Data ext
DATA_FILE = "data"

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
INITIAL_SCALAR_FITNESS = -999.0

OPTIMIZATION_TYPE_EVOLUTION = "EO"
OPTIMIZATION_TYPE_LINEAR = "LO"

MAX_STORING_TIME_OPTIMIZATION_TASKS = 20  # in days

DATE_FORMAT = "%Y-%m-%d"

MODFLOW_EXE = "mf2005"
