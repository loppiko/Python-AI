from datetime import time, datetime


class WorkTime:
    def __init__(self, startTime: time = time(0), endTime: time = time(0)) -> None:
        self.startTime = startTime
        self.endTime = endTime
        

    @staticmethod
    def convertFromString(inputString: str) -> 'WorkTime':
        separetedStrings = inputString.split("-")

        startTime = datetime.strptime(separetedStrings[0], '%H:%M').time()
        endTime = datetime.strptime(separetedStrings[1], '%H:%M').time()

        return WorkTime(startTime, endTime)

    
    def isBetween(self, otherWorkTime: 'WorkTime') -> bool:
        if (self.startTime <= self.endTime and otherWorkTime.startTime <= otherWorkTime.endTime):
            return self.startTime >= otherWorkTime.startTime and self.endTime <= otherWorkTime.endTime
        elif (self.startTime >= self.endTime and otherWorkTime.startTime <= otherWorkTime.endTime):
            return False
        elif (self.startTime >= self.endTime and otherWorkTime.startTime >= otherWorkTime.endTime):
            if (self.startTime >= otherWorkTime.startTime):
                return self.endTime < otherWorkTime.endTime
            else:
                return self.startTime < otherWorkTime.endTime and self.endTime < otherWorkTime.endTime
        else:
            return self.startTime >= otherWorkTime.startTime


    def notOverlaping(self, otherWorkTime: 'WorkTime') -> bool:
        if (self.startTime <= self.endTime and otherWorkTime.startTime <= otherWorkTime.endTime):
            return self.endTime <= otherWorkTime.startTime or self.startTime >= otherWorkTime.endTime
        elif (self.startTime >= self.endTime and otherWorkTime.startTime <= otherWorkTime.endTime):
            if (self.startTime >= otherWorkTime.startTime):
                return otherWorkTime.startTime >= self.endTime and otherWorkTime.endTime <= self.startTime
            else:
                return False
        elif (self.startTime <= self.endTime and otherWorkTime.startTime >= otherWorkTime.endTime):
            if (self.startTime <= otherWorkTime.startTime):
                return otherWorkTime.startTime >= self.endTime and otherWorkTime.endTime < self.startTime
            else:
                return False
        else:
            return False
        

    def __str__(self) -> str:
        return f"{self.startTime.strftime('%H:%M')}-{self.endTime.strftime('%H:%M')}"
    



class Worker:
    def __init__(self, workerName: str = "", profession: str = "", workHours: list[WorkTime] = [], hourlySalary: float = 0.0) -> None:
        self.workerName = workerName
        self.profession = profession
        self.workHours = workHours
        self.hourlySalary = hourlySalary
        self.alreadyAssigned = {}

    @staticmethod
    def convertFromDictionary(inputDictionary: dict['Worker']) -> 'Worker':
        workHours = []

        for key in list(inputDictionary["workHours"].keys()):
            workHours.append(WorkTime().convertFromString(inputDictionary["workHours"][key]))

        return Worker(inputDictionary["workerName"], inputDictionary["profession"], workHours, inputDictionary["hourlySalary"])


    def checkProfession(self, inputProfession) -> bool:
        return inputProfession == self.profession
    

    def checkHoursAvaibility(self, dayNumber: int, eventWorkTime: WorkTime) -> bool:
        return eventWorkTime.isBetween(self.workHours[dayNumber])
    

    def assignWorkTime(self, dayNumber: int, workTime: WorkTime) -> None:
        if (str(dayNumber) in self.alreadyAssigned):
            self.alreadyAssigned[dayNumber].append(workTime)
        else:
            self.alreadyAssigned[dayNumber] = [workTime]


    def isBusy(self, dayNumber: int, workTime: WorkTime) -> bool:
        if (str(dayNumber) not in self.alreadyAssigned):
            return False
        
        for workTime in self.alreadyAssigned[str(dayNumber)]:
            if (not workTime.notOverlaping(workTime)):
                return True
        
        return False


    def __toDict__(self) -> dict:
        return {
            "workerName": self.workerName,
            "profession": self.profession,
            "workHours": self.workHours,
            "hourlySalary": self.hourlySalary
        }