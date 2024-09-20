import model, graph, pygame, math, itertools

pygame.init()

""" Just colors """
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
DARK_RED = (200, 0, 0)
BLUE = (0, 0, 255)
DARK_BLUE = (0, 0, 200)

""" Global GUI parameters """
WIDTH = 1600
HEIGHT = 800
FRAME_RATE = 60
CAPTION = "Deterministic Model Routing Simulation"
ADVERSARY = "adversary.png"
ALICE = "alice.png"
BOB = "bob.png"

""" Global GUI variables """
display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(CAPTION)
clock = pygame.time.Clock()

""" Load all images """
adversary = pygame.image.load(ADVERSARY)
alice = pygame.image.load(ALICE)
bob = pygame.image.load(BOB)


class Simulation(graph.NetworkGraph):

    def __init__(self, num_nodes=12, num_paths=1, **kwargs):
        """Model is used to generate curiosity and collaboration and also perform the routing algorithm"""
        self.model = model.ProbabilisticModel
        display.fill(WHITE)
        """ Store the variable arguments for later use """
        self.kwargs = kwargs
        self.at_least_num_paths = num_paths
        graph.NetworkGraph.__init__(self, num_nodes)
        self.__reset_graph__()
        self.__update_model__()
        """ Set up colors for adversaries """
        self.__update_colors__()

        """ rectangle_list contains list of rectangles to update """
        self.rectangle_list = []

        """ event_handler is a dictionary which maps event type to list of event handlers """
        self.event_handlers = {}

        """ Set up the simulation using start_simulation method, currently set to an empty generator """
        self.simulation = iter(())
        """ Just set a default value, may not be used """
        self.obj_fn = model.ProbabilisticModel.ObjectFunction.NO_NODE_BREAK_SECRET

        # """ Draw the buttons on initialization """
        # self.draw_buttons()

    def __reset_graph__(self):
        """ensure if possible 'num_paths', and update layout points"""
        self.ensure_at_least_n_paths(self.at_least_num_paths)
        self.calculate_layout_points()

    def __update_model__(self):
        """update curiosity and collaboration with new values"""
        self.curiosity = self.model.random_curiosity(self.num_nodes, **self.kwargs)
        self.collaboration = self.model.random_collaboration(self.num_nodes, **self.kwargs)

    @staticmethod
    def create_colors(n):
        """Number of possible values for individual channels"""
        p = math.ceil(math.pow(n + 1, 1 / 3))
        """ Possible values for individual channels """
        c = [math.ceil((255 * i) / (p - 1)) for i in range(0, p)]

        def color_priority(color):
            """Define a priority for colors"""
            return sum(c if c > 0 else 512 for c in color)

        """ Sort color based on priority """
        return sorted(list(itertools.product(c, c, c)), key=color_priority, reverse=True)[:n]
    
    def __update_colors__(self):
        """This function generates colors for adversarial nodes"""
        """ Update the colors, a color for each adversary """
        self.colors = Simulation.create_colors(self.num_adversaries)

    @staticmethod
    def color_rect_with_value(surface, rect, color_value):
        """Fill the given rect on the surface with a given color_value according to color scale and
        also label the value for more readability"""
        pygame.draw.rect(surface, Simulation.color_scale(color_value), rect)
        """Obtain font size such that color value fits in rect"""
        font_size = min(rect.height / 2, rect.width / 3)
        Simulation.draw_text(surface, "%0.2f" % color_value, font_size, rect.center, text_color=WHITE)

    @staticmethod
    def create_color_image(image: pygame.Surface, target_color, src_color=WHITE, threshold=(10, 10, 10)):
        """This function finds source color (color within threshold of src_color) in image
        and replaces it with target color and returns the new image"""
        new_image = image.copy()
        pygame.transform.threshold(
            dest_surface=new_image,
            surface=image,
            search_color=src_color,
            threshold=threshold,
            set_color=target_color,
            inverse_set=True,
        )
        return new_image

    @staticmethod
    def resize_image(image: pygame.Surface, width, height):
        """This function returns a re-sized version of given image"""
        return pygame.transform.scale(image, (int(width), int(height)))

    @staticmethod
    def create_adversary(target_color, width, height):
        """Create an adversary image of given color"""
        resized_image = Simulation.resize_image(adversary, width, height)
        return Simulation.create_color_image(resized_image, target_color)
    
    @staticmethod
    def draw_surface(dst_surface, src_surface, left, top):
        """This function draws the src_surface into the dst_surface at given position"""
        dst_surface.blit(src_surface, (int(left), int(top)))

    def invalidate_rect(self, surf, rect):
        """This function is used to draw an updated rect"""
        display.blit(surf, rect)
        self.rectangle_list.append(rect)

    def update_1d_metric(self, left, top, width, height, values, label):
        """This function is used to update an 1D metric of adversaries on the GUI
        It has the format of metrics, adversaries, label
        """
        """ Note: values must includes source and end point values, which are ignored """
        assert len(values) == self.num_nodes
        surface, rectangle = Simulation.construct_surface(left, top, width, height, False)
        num_adversaries = self.num_adversaries
        adversary_width, adversary_height = width / num_adversaries, height / 3
        for i in range(num_adversaries):
            rect = Simulation.rect(adversary_width * i, 0, adversary_width, adversary_height)
            Simulation.color_rect_with_value(surface, rect, values[i + 1])
        """ Draw the adversaries """
        for i in range(num_adversaries):
            tmp_adversary = self.create_adversary(self.colors[i], adversary_width, adversary_height)
            Simulation.draw_surface(surface, tmp_adversary, adversary_width * i, adversary_height)
        Simulation.draw_text(surface, label, adversary_height / 2, (width / 2, height * 5 / 6))
        self.invalidate_rect(surface, rectangle)

    @staticmethod
    def draw_text(
        surf: pygame.Surface,
        text: str,
        font_size,
        center=None,
        position=None,
        font_family="freesansbold.ttf",
        text_color=BLACK,
    ):
        """Draws a given text on a surface with given font size at given position on the surface with given color"""
        """Create the font and render text with font with given color"""
        font = pygame.font.Font(font_family, int(font_size))
        text_surf = font.render(text, True, text_color)
        text_rect = text_surf.get_rect()
        if center:
            """If center is specified then update the center of rect"""
            text_rect.center = (int(center[0]), int(center[1]))
        if position:
            """If start position is specified, the update left and top"""
            text_rect.left, text_rect.top = int(position[0]), int(position[1])
        surf.blit(text_surf, text_rect)

    @staticmethod
    def rect(left, top, width, height):
        """Constructs a Rect using given positions which may possibly be float values"""
        return pygame.Rect(int(left), int(top), int(width), int(height))

    @staticmethod
    def construct_surface(left, top, width, height, return_dimensions=True):
        """Construct a surface and fill it with a default color"""
        rectangle = Simulation.rect(left, top, width, height)
        surface = pygame.Surface((rectangle.width, rectangle.height))
        surface.fill(WHITE)
        if return_dimensions:
            """Return dimensions as well if requested"""
            return surface, rectangle, rectangle.width, rectangle.height
        return surface, rectangle


    def update_curiosity(self, curiosity):
        self.update_1d_metric(0, HEIGHT * 3 / 4, WIDTH / 4, HEIGHT / 8, curiosity, "CURIOSITY")

    def update_collaboration(self, collaboration, diagonal=1):
        assert diagonal == 0 or diagonal == 1
        """ This function updates the collaboration values on the GUI """
        surface, rectangle, width, height = Simulation.construct_surface(0, HEIGHT / 4, WIDTH / 4, HEIGHT / 2)
        num_adversaries = self.num_adversaries
        adversary_width, adversary_height = width / (num_adversaries + 1), height / (num_adversaries + 2)
        for i in range(0, num_adversaries):
            tmp_adversary = self.create_adversary(self.colors[i], adversary_width, adversary_height)
            Simulation.draw_surface(surface, tmp_adversary, 0, adversary_height * i)
        for j in range(0, num_adversaries):
            tmp_adversary = self.create_adversary(self.colors[j], adversary_width, adversary_height)
            Simulation.draw_surface(
                surface, tmp_adversary, adversary_width * (j + 1), adversary_height * num_adversaries
            )
        for i in range(0, num_adversaries):
            for j in range(0, i + diagonal):
                rect = Simulation.rect(
                    adversary_width * (j + 1), adversary_height * i, adversary_width, adversary_height
                )
                """ Assuming adversaries in collaboration matrix start from 1, ie, 0 is start-point"""
                Simulation.color_rect_with_value(surface, rect, collaboration[i + 1, j + 1])

        """ Label the surface """
        self.draw_text(surface, "COLLABORATION", adversary_height / 2, (width / 2, height - adversary_height / 2))
        """ Draw the surface onto the display during the next update """
        self.invalidate_rect(surface, rectangle)

    @staticmethod
    def color_scale(value: float):
        """This function maps a float value to a color. This function is used to visualize floats between 0 and 1"""
        assert 0 <= value <= 1.0
        color = (int(255 * value), int(255 - 255 * value), 0)
        return color
    

    @staticmethod
    def draw_line(surface, color, start_x, start_y, end_x, end_y, width=1):
        """A wrapper function around pygame.draw.line"""
        pygame.draw.line(surface, color, (int(start_x), int(start_y)), (int(end_x), int(end_y)), int(width))


    def draw_color_scale(self, resolution=1000):
        surface, rectangle, width, height = Simulation.construct_surface(0, HEIGHT * 7 / 8, WIDTH / 4, HEIGHT / 8)
        for i in range(resolution):
            color = Simulation.color_scale(i / resolution)
            x, line_width = width * i / resolution, width / resolution + 1
            Simulation.draw_line(surface, color, x, 0, x, height / 2, line_width)
        ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for tick in ticks:
            Simulation.draw_text(surface, "%0.1f" % tick, height / 8, (width * tick, height * 3 / 4))
        self.invalidate_rect(surface, rectangle)
    
    def start_simulation(self, obj_fn=model.ProbabilisticModel.ObjectFunction.NO_NODE_BREAK_SECRET):
        """Set up the starting GUI and run simulation_next to start the simulation"""
        self.update_curiosity(self.curiosity)
        self.update_collaboration(self.collaboration)
        self.draw_color_scale()
        self.obj_fn = obj_fn
        # TODO:
        self.simulation = model.simulator(
            self.num_nodes, self.reduced_paths, self.curiosity, self.collaboration, obj_fn
        )
        print("START SIMULATION")
        # self.simulation_next()


s = Simulation(12, 3)
s.start_simulation()
# s.main_loop()