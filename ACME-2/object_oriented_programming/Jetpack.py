from Backpack import Backpack

class Jetpack(Backpack):

    def __init__(self, color='silver', name='jetpack', max_size=2, fuel=10):
        self.color = color
        self.name = name
        self.max_size = max_size
        self.fuel = fuel
        self.contents = []

    def fly(self, fuel_to_be_burned):
        if fuel_to_be_burned > self.fuel:
            print "Not enough fuel!"
        else:
            self.fuel -= fuel_to_be_burned

    def dump(self):
        """Empties the contents of a jetpack and fuel."""
        self.contents[:] = []
        self.fuel = 0
