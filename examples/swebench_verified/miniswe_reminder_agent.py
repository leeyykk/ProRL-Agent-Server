"""Mini-SWE default agent with the same turn reminders used by SkyRL."""

from minisweagent.agents.default import DefaultAgent


class DefaultAgentWithReminder(DefaultAgent):
    """Append SkyRL's remaining-turn reminder after every observation."""

    def get_observation(self, response: dict) -> dict:
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(
            self.config.action_observation_template, output=output
        )
        remaining = self.config.step_limit - self.model.n_calls

        if remaining == 1:
            observation = (
                f"{observation}\nREMINDER: You only have 1 turn left. "
                "Please provide the final answer"
            )
        elif remaining > 1:
            observation = (
                f"{observation}\nREMINDER: You have {remaining} turns left "
                "to arrive at the solution."
            )

        self.add_message("user", observation)
        return output
