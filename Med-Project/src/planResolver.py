import pygad

from myTypes import projectTypes
from other import utils

import plotly.express as px
import pandas as pd


### REWARDS
STARTING_REWARD = 500
STARTING_TASK_REWARD = 50

### PENALTIES
WRONG_HOURS_PENALTY = 300
WRONG_PROFESSION_PENALTY = 150
ALREADY_BUSY_PENALTY = 150

### MODYFIERS
SALARY_MULTIPLIER = 0.5



class PlanResolver():
    def __init__(self, professionFilePath: str, workersFilePath: str, scheduleFilePath: str) -> None:
        self.professions: dict = utils.readJson(professionFilePath)
        self.workers: list[dict] = utils.readJson(workersFilePath)
        self.schedule: list[dict] = self.pythonReduce(utils.readJson(scheduleFilePath))


    def initializePygad(self):
        self.pygad = pygad.GA(
            num_generations=200,
            num_parents_mating=3,
            fitness_func=self.fitnessFunc,
            sol_per_pop=50,
            num_genes=len(self.schedule),
            init_range_low=0,
            init_range_high=2,
            gene_space=[workerIndex for workerIndex in range(len(self.workers))],
        )

        self.pygad.run()

        self.solution, self.solution_fitness, _ = self.pygad.best_solution()

        self.pygad.plot_fitness(save_dir="answers.png")

        print(f"\n---------Results:---------\n")
        print(f"Solution:\n{self.solution}\n")
        print(f"FINAL FITNESS: {self.solution_fitness}")

        self.drawPlan()


    def pythonReduce(self, inputDictionary: list[dict]) -> list[dict]:
        result = []

        for dayScheduled in inputDictionary:
            currDayNumber = dayScheduled["day"]

            for event in dayScheduled["events"]:
                event["day"] = currDayNumber
                result.append(event)

        return result


    def checkSolution(self, solution):
        
        totalReward = STARTING_REWARD

        for index, workerNumber in enumerate(solution):
            worker = projectTypes.Worker.convertFromDictionary(self.workers[int(workerNumber)])
            totalReward += self.checkSolutionQuality(worker, self.schedule[index])

        return totalReward
             
        #TODO return number which represents quality of solution !DONE
        #TODO Check if worker is not already been choosen in specified hours !DONE
        #TODO Specify reward according to salary !DONE


    def checkSolutionQuality(self, worker: projectTypes.Worker, event: dict) -> int:
        reward = STARTING_TASK_REWARD

        eventWorkTime = projectTypes.WorkTime.convertFromString(event["time"])

        if (not worker.checkHoursAvaibility(event["day"], eventWorkTime)):
            reward -= WRONG_HOURS_PENALTY
        if (not worker.checkProfession(event["role"])):
            reward -= WRONG_PROFESSION_PENALTY
        if (worker.isBusy(event["day"], eventWorkTime)):
            reward -= ALREADY_BUSY_PENALTY
    
        worker.assignWorkTime(event["day"], eventWorkTime)

        reward -= worker.hourlySalary * SALARY_MULTIPLIER

        return reward


    def fitnessFunc(self, _1, solution, _2):
        return self.checkSolution(solution)


    def drawPlan(self):
        dataFrames = []

        for eventIndex, workerIndex in enumerate(self.solution):
            dataFrames.append(
                dict(Task=f"{self.schedule[eventIndex]['activity']}", 
                     Start=f"2024-01-0{self.schedule[eventIndex]['day'] + 1} {self.schedule[eventIndex]['time'].split('-')[0]}", 
                     Finish=f"2024-01-0{self.schedule[eventIndex]['day'] + 1} {self.schedule[eventIndex]['time'].split('-')[1]}", 
                     Name=f"{self.workers[int(workerIndex)]['workerName']}\n({self.workers[int(workerIndex)]['profession']})"))


        df = pd.DataFrame(dataFrames)


        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Name", text="Task")
        fig.update_yaxes(autorange="reversed")
        fig.show()
