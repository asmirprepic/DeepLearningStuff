import tensorflow as tf

fair_pros = tf.ones(2)/2 # Creating a vector with probabilities 0.5,0.5
tfd.Multinomial(100,fair_probs).sample() ## Simulating 100 draws from the distribution 0.5,0.5
tfd.Multinomial(100,fair_probs).sample()/100 ## Calculating the simulated probabilities

