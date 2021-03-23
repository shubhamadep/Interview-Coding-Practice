import enum
from abc import ABC

class TrafficLightState(enum.Enum):
    GREEN, YELLOW, RED = 1, 2, 3

class VehicleType(enum.Enum):
    CAR, VAN, CYCLE = 1, 2, 3

class TrafficLight():
    def __init__(self):
        self._status = TrafficLightState.RED

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        self._status = status


class ParkingSpot(ABC):
    def __init__(self, type, free):
        self._type = type
        self._free = free
        self._vehicle = None

    def is_free(self):
        return self._free

    def assign_vehicle(self, vehicle):
        self._vehicle = vehicle
        self._free = False

    def remove_vehicle(self):
        self._vehicle = None
        self._free = True

class ParkingFloor():
    def __init__(self):
        self._capacity = 0

    @property
    def capacity(self):
        return self._capacity

    @capacity.setter
    def capacity(self, capacity):
        if capacity < 0:
            raise Exception("Parking cannot have negative capacity!.")
        self._capacity = capacity

    def parking_available(self):
        return self._capacity > 0

class Vehicle():
    def __init__(self, licenseNumber, vehicleType):
        self.license = licenseNumber
        self.vehicleType = vehicleType
        self.ticket_number = None

    def assign_ticket(self, ticket_number):
        self.ticket_number = ticket_number

class Car(Vehicle):
    def __init__(self, license_number, ticket=None):
        super().__init__(license_number, VehicleType.CAR)

class Van(Vehicle):
    def __init__(self, license_number, ticket=None):
        super().__init__(license_number, VehicleType.VAN)


Pf = ParkingFloor()
Pf.capacity = 10
print(Pf.capacity)

tl = TrafficLight()
tl.status = TrafficLightState.GREEN
print(tl.status)

