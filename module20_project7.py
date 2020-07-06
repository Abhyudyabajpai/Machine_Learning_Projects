import numpy as np

#P(answers correctly | knows the material)* P(knows the material) OR (+)
#P(answerws correctly | does not know the material)* P(does not know the material)

#P(A|B) =P(B|A)*P(A)/P(B)

p_knows_the_material_given_answers_correctly = 0.85*.6/0.59
print(p_knows_the_material_given_answers_correctly)
