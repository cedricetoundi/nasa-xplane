from NL2STL.openai_func import *
import torch

model_name = 'gpt-4'

class LLMCheck:

    def __init__(self, agent):
        self.agent = agent

    def get_state(self):

        """ Get the states to be checked and see whether or not they went 
            above a user defined limit during the entire experiment
        """
        state = self.agent.get_observation()
        self.speed = torch.tensor([[[state['groundspeed']]]])

        return self.speed

    def code_generation(self, user_input):
        """
            Generates a string of the code defining and evaluating the STL based on a command or user input

        """

        prompt = """

        You are tasked with using the STLCG package to transform natural language propositions into Signal Temporal Logic (STL) code and evaluate their robustness. Below are detailed instructions along with example code. Your response should also be in a code string format, which can be later converted to actual code using the eval function.

        Instructions:

        1. Import the necessary components from the STLCG package, which includes operators such as Always, Expression, Negation, Implies, And, Or, Until, and Then.

        2. Define a natural language proposition you want to transform into STL code. For example: "If the temperature is greater than 30 degrees, then turn on the air conditioning."

        3. Transform the natural language proposition into STL code using the components provided by the STLCG package. You may need to define signals and expressions based on the proposition.

        4. Evaluate the STL formula to calculate the robustness. Make sure to provide the necessary inputs required for the evaluation.

        5. Check if the proposition is violated or not based on the robustness result. Print an appropriate message.


        The code must be structured exactly like this Example:

        # Import the necessary components
        from stlcg.src.stlcg import Always, Expression, Implies
        import torch

        # Define the natural language proposition
        natural_language_proposition = "If the temperature is greater than 30 degrees, then turn on the air conditioning."

        # Define signals and expressions
        threshold = 30
        temperature = self.temperature (It can be self.speed or self.position or self.latitude)
        temperature_exp = Expression('temperature', temperature)
        threshold_exp = Expression('threshold', threshold)
       
        
        # Create the STL formula
        temperature_condition = (temperature_exp > threshold_exp)
    
        stl_formula = Implies(subformula1=temperature_condition)

        # Define input signals
        inputs = (temperature_exp) # Do not forget to define the expression above as the robustness uses expressions

        # Calculate the robustness
        robustness_result = stl_formula.robustness(inputs, pscale=1, scale=-1)

        # Check if the proposition is violated or not
        if robustness_result < 0:
            print('The proposition is violated: Turn on the air conditioning.')
        else:
            print('The proposition is not violated: Do not turn on the air conditioning.')

        # MAKE SURE THAT YOUR CODE FOLLOWS THIS EXACT PATTERN OTHERWISE IT WON'T WORK
        # Do not use backslash n to go to the next line

        

        """


        user_prompt_1 = prompt + '\n\nInput: \n' + user_input + '\n\nOuput: '


        if model_name == 'gpt-4':
            Output = GPT_response_first_round(user_prompt_1, 'gpt-4')
        elif model_name == 'gpt-3':
            Output = GPT_response_GPT_3(user_prompt_1)
        
        return Output

    def command_evaluation(self, code):
        """
            Executes the python code within the triple-quoted string
        
        """

        exec(code)
            
        