from other import utils
from myTypes import projectTypes
from planResolver import PlanResolver


professionFilePath = "database/professionsList.json"
scheduleFilePath = "database/schedule.json"
workersFilePath = "database/workers.json"

resolver = PlanResolver(professionFilePath, workersFilePath, scheduleFilePath)
resolver.initializePygad()