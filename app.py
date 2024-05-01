from flask import Flask, request, jsonify
import random as rnd
from datetime import datetime, timedelta

import prettytable

app = Flask(__name__)

# Constants for Event Scheduling
NUMB_OF_ELITE_EVENTS = 2
MUTATION_RATE = 0.2
TOURNAMENT_SELECTION_SIZE_EVENTS = 5
EVENT_DURATION = 1  # in hours

# Define global variables to store scheduling data
data = None
EVENT_START_TIME = None
EVENT_END_TIME = None
TIME_SLOT_INTERVAL = None
POPULATION_SIZE = None
MAX_GENERATIONS = None

class Event:
    def __init__(self, id, name):
        self._id = id
        self._name = name
        self._room = None
        self._time_slot = None

    # Define getters for ID, name, room, and time slot
    def get_id(self):
        return self._id

    def get_name(self):
        return self._name

    def get_room(self):
        return self._room

    def get_time_slot(self):
        return self._time_slot

    # Define setters for room and time slot
    def set_room(self, room):
        self._room = room

    def set_time_slot(self, time_slot):
        self._time_slot = time_slot

    # Define string representation of the event
    def __str__(self):
        return f"{self._name}, Room: {self._room.get_number()}, Time: {self._time_slot.strftime('%I:%M %p') if self._time_slot else 'N/A'}"

# Room class represents a room with a number and availability schedule
# Room class represents a room with a number and availability schedule
class Room:
    def __init__(self, number, availability_schedule):
        self._number = number
        self._availability_schedule = availability_schedule

    def get_number(self):
        return self._number

    def is_available(self, time_slot):
        index = int((time_slot - EVENT_START_TIME) / TIME_SLOT_INTERVAL)
        return self._availability_schedule[index]

class Data:
    def __init__(self, num_rooms, room_names, event_names, start_time, end_time, room_availability):
        self._rooms = []
        self._events = []

        # Calculate time-related variables
        global EVENT_START_TIME, EVENT_END_TIME, TIME_SLOT_INTERVAL, POPULATION_SIZE, MAX_GENERATIONS
        EVENT_START_TIME = datetime.strptime(start_time, '%I:%M %p')
        EVENT_END_TIME = datetime.strptime(end_time, '%I:%M %p')
        TIME_SLOT_INTERVAL = timedelta(hours=1)
        POPULATION_SIZE = 20
        MAX_GENERATIONS = 200

        # Create Room objects
        for i in range(num_rooms):
            self._rooms.append(Room(room_names[i], room_availability[i]))

        # Create Event objects
        for i in range(len(event_names)):
            self._events.append(Event(i, event_names[i]))

    def get_rooms(self):
        return self._rooms

    def get_events(self):
        return self._events

# ScheduleEvents class represents a schedule of events
class ScheduleEvents:
    def __init__(self, data):
        self._data = data
        self._events = []
        self._num_of_conflicts = 0
        self._fitness = -1
        self._event_num = 0
        self._is_fitness_changed = True

    def get_events(self):
        self._is_fitness_changed = True
        return self._events

    def get_num_of_conflicts(self):
        return self._num_of_conflicts

    def get_fitness(self):
        if self._is_fitness_changed:
            self._fitness = self.calculate_fitness()
            self._is_fitness_changed = False

        return self._fitness

    def initialize(self):
        for i in range(len(self._data.get_events())):
            new_event = Event(self._event_num, self._data.get_events()[i].get_name())
            self._event_num += 1
            new_event.set_time_slot(self.get_random_time_slot())
            new_event.set_room(self.get_available_room(new_event.get_time_slot()))
            self._events.append(new_event)

        return self

    def calculate_fitness(self):
        self._num_of_conflicts = 0
        events = self.get_events()

        for i in range(len(events)):
            if not events[i].get_room().is_available(events[i].get_time_slot()):
                self._num_of_conflicts += 1

            for j in range(len(events)):
                if j >= i:
                    if events[i].get_time_slot() == events[j].get_time_slot() and events[i].get_id() != events[j].get_id():
                        if events[i].get_room() == events[j].get_room():
                            self._num_of_conflicts += 1

        return 1 / (1.0 * (self._num_of_conflicts + 1))

    def get_random_time_slot(self):
        current_time = EVENT_START_TIME
        available_time_slots = []

        while current_time < EVENT_END_TIME:
            available_time_slots.append(current_time)
            current_time += TIME_SLOT_INTERVAL

        chosen_time_slot = rnd.choice(available_time_slots)
        return chosen_time_slot

    def get_available_room(self, time_slot):
        available_rooms = [room for room in self._data.get_rooms() if room.is_available(time_slot)]

        if not available_rooms:
            # Handle the case where no available rooms are found for the given time slot
            raise Exception("No available rooms for the given time slot")

        return rnd.choice(available_rooms)

    def __str__(self) -> str:
        return_value = ''
        for i in range(len(self._events) - 1):
            return_value += str(self._events[i]) + ', '

        return_value += str(self._events[len(self._events) - 1])

        return return_value

# PopulationEvents class represents a population of schedules
class PopulationEvents:
    def __init__(self, size, data):
        self._size = size
        self._data = data
        self._schedules = []
        for i in range(size):
            self._schedules.append(ScheduleEvents(data).initialize())

    def get_schedules(self):
        return self._schedules

# GeneticAlgorithmEvents class represents the genetic algorithm for evolving schedules
class GeneticAlgorithmEvents:
    def evolve(self, population):
        return self._mutate_population(self._crossover_population(population))

    def _crossover_population(self, pop):
        crossover_pop = PopulationEvents(0, pop._data)
        for i in range(NUMB_OF_ELITE_EVENTS):
            crossover_pop.get_schedules().append(pop.get_schedules()[i])

        i = NUMB_OF_ELITE_EVENTS

        while i < POPULATION_SIZE:
            schedule1 = self._select_tournament_population(pop).get_schedules()[0]
            schedule2 = self._select_tournament_population(pop).get_schedules()[0]
            crossover_pop.get_schedules().append(self._crossover_schedule(schedule1, schedule2))

            i += 1

        return crossover_pop

    def _mutate_population(self, population):
        for i in range(NUMB_OF_ELITE_EVENTS, POPULATION_SIZE):
            self._mutate_schedule(population.get_schedules()[i])

        return population

    def _crossover_schedule(self, schedule1, schedule2):
        crossover_schedule = ScheduleEvents(schedule1._data).initialize()
        for i in range(0, len(crossover_schedule.get_events())):
            if rnd.random() > 0.5:
                crossover_schedule.get_events()[i] = schedule1.get_events()[i]
            else:
                crossover_schedule.get_events()[i] = schedule2.get_events()[i]

        return crossover_schedule

    def _mutate_schedule(self, mutate_schedule):
        schedule = ScheduleEvents(mutate_schedule._data).initialize()
        for i in range(len(mutate_schedule.get_events())):
            if MUTATION_RATE > rnd.random():
                mutate_schedule.get_events()[i] = schedule.get_events()[i]
        return mutate_schedule

    def _select_tournament_population(self, pop):
        tournament_pop = PopulationEvents(0, pop._data)
        i = 0
        while i < TOURNAMENT_SELECTION_SIZE_EVENTS:
            tournament_pop.get_schedules().append(pop.get_schedules()[rnd.randrange(0, POPULATION_SIZE)])
            i += 1

        tournament_pop.get_schedules().sort(key=lambda x: x.get_fitness(), reverse=True)

        return tournament_pop

# DisplayMgrEvents class handles the display of available data, generations, and schedules
class DisplayMgrEvents:
    def print_available_data(self):
        print("> All Available Data")
        self.print_rooms()
        self.print_events()

    def print_rooms(self):
        rooms = data.get_rooms()
        available_rooms_table = prettytable.PrettyTable(['Room', 'Availability Schedule'])

        for i in range(len(rooms)):
            room_schedule = ['Available' if slot else 'Unavailable' for slot in rooms[i]._availability_schedule]
            available_rooms_table.add_row([rooms[i].get_number(), room_schedule])

        print(available_rooms_table)

    def print_events(self):
        available_events_table = prettytable.PrettyTable(['Event #', 'Event Name'])
        events = data.get_events()

        for i in range(len(events)):
            available_events_table.add_row([events[i].get_id(), events[i].get_name()])

        print(available_events_table)

    def print_generation(self, population):
        table1 = prettytable.PrettyTable(['Schedule #', 'Fitness', '# of conflicts', 'Events'])
        schedules = population.get_schedules()

        for i in range(len(schedules)):
            table1.add_row([str(i + 1), round(schedules[i].get_fitness(), 3), schedules[i].get_num_of_conflicts(),
                            schedules[i].__str__()])

        print(table1)

    def print_schedule_as_table(self, schedule):
        table1 = prettytable.PrettyTable(['Event #', 'Event Name', 'Room', 'Time Slot'])
        events = schedule.get_events()

        for i in range(len(events)):
            table1.add_row(
                [events[i].get_id(), events[i].get_name(), events[i].get_room().get_number()
                    if events[i].get_room() is not None else 'N/A',
                 events[i].get_time_slot().strftime('%I:%M %p') if events[i].get_time_slot() is not None else 'N/A']
            )

        print(table1)

@app.route('/schedule', methods=['POST'])
def schedule_events():
    global data
    json_data = request.json
    data = Data(
        num_rooms=json_data['numRooms'],
        room_names=json_data['roomNames'],
        event_names=json_data['eventNames'],
        start_time=json_data['startTime'],
        end_time=json_data['endTime'],
        room_availability=json_data['roomAvailability']
    )

    # Run scheduling algorithm
    population = PopulationEvents(POPULATION_SIZE, data)
    genetic_algorithm = GeneticAlgorithmEvents()

    generation_number = 0
    while population.get_schedules()[0].get_fitness() != 1.0 and generation_number < MAX_GENERATIONS:
        generation_number += 1
        population = genetic_algorithm.evolve(population)

    # Get the best schedule
    best_schedule = population.get_schedules()[0]

    # Return the best schedule as JSON response
    return jsonify({
        "bestSchedule": {
            "fitness": best_schedule.get_fitness(),
            "events": [{"eventName": event.get_name(), "room": event.get_room().get_number(),
                        "timeSlot": event.get_time_slot().strftime('%I:%M %p')}
                       for event in best_schedule.get_events()]
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
