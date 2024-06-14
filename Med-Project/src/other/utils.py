from myTypes import projectTypes

from datetime import timedelta, time, datetime
from faker import Faker
import random
import json


ALL_AVAIBLE_MINUTES = [0, 15, 30, 45]
AVAIBLE_HOUR_SALARY = (30, 120)
WORK_TIME = time(8, 0)


def readJson(jsonPath: str) -> dict:
    with open(jsonPath, "r") as jsonFile:
        data = json.load(jsonFile)
    return data
    

def randomTime() -> time:
    hour = random.randint(0, 23)
    minute = random.choice(ALL_AVAIBLE_MINUTES)
    return time(hour, minute)


def generateWorkTime() -> projectTypes.WorkTime:
    startTime = randomTime()

    startDatetime = datetime.combine(datetime.min, startTime)
    addedTime = timedelta(hours=WORK_TIME.hour, minutes=WORK_TIME.minute)
    new_datetime = startDatetime + addedTime
    
    if new_datetime.day > 1:
        new_datetime -= timedelta(days=1)
    
    return projectTypes.WorkTime(startTime, new_datetime.time())



def generateWorkers(numberOfWorkers: int, workersDatabaseLocation: str, professionList: list[str]) -> None:
    result: list[dict] = []
    fakeNames = Faker()

    for _ in range(numberOfWorkers):
        currName = fakeNames.name()
        currProfession = random.choice(professionList)

        currWorkHours = {f"{index}": generateWorkTime().__str__() for index in range(7)}

        currHourSalary = random.randint(AVAIBLE_HOUR_SALARY[0], AVAIBLE_HOUR_SALARY[1])

        result.append(projectTypes.Worker(currName, currProfession, currWorkHours, currHourSalary).__toDict__())


    jsonString = json.dumps(result, indent=4) 

    with open(workersDatabaseLocation, "w") as jsonFile:
        jsonFile.write(jsonString)