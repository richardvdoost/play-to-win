from agents.agent import Agent


class HumanAgent(Agent):
    def act(self, observation, reward, done, legal_actions=None):
        if self.game.viewer and self.game.viewer.pygame:
            viewer = self.game.viewer
        else:
            raise Exception(
                f"Pygame is required for {self.__class__}! "
                "Please render the environment using mode=human"
            )

        while not viewer.exited and not done:

            for event in viewer.pygame.event.get():
                if event.type == viewer.pygame.QUIT:
                    viewer.exited = True
                    viewer.pygame.quit()
                    return None

            action = viewer.get_pygame_move()

            if action is not None:
                return action

            viewer.clock.tick(60)
