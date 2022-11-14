class Bayes:
    def __init__(self, hypotheses: list, priors: list, observations: list, likelihoods: list):
        """
        Constructs the attributes for the Bayes class
        :param hypotheses: List of hypotheses
        :param priors: List of prior probabilities of hypotheses
        :param observations: List of observations
        :param likelihoods: A nested list with shape [#hypotheses, #observations] with likelihoods of each observation
        given each hypothesis
        """
        self.hypotheses = hypotheses
        self.priors = priors
        self.observations = observations
        self.likelihoods = likelihoods
        assert len(self.priors) == len(self.hypotheses), 'Mismatch between the shape of hypotheses and priors'
        assert len(self.likelihoods) == len(self.hypotheses), 'Mismatch between the shape of hypotheses and likelihoods'
        for lik in self.likelihoods:
            assert len(lik) == len(self.observations), 'Mismatch between the shape of hypotheses and likelihoods'

    def likelihood(self, observation, hypothesis):
        return self.likelihoods[self.hypotheses.index(hypothesis)][self.observations.index(observation)]

    def norm_constant(self, observation):
        c = 0
        for hyp in self.hypotheses:
            c += self.likelihoods[self.hypotheses.index(hyp)][self.observations.index(observation)] *\
                 self.priors[self.hypotheses.index(hyp)]
        return c

    def single_posterior_update(self, observation):
        posteriors = []
        for hyp in self.hypotheses:
            posteriors.append(
                self.priors[self.hypotheses.index(hyp)] * self.likelihood(observation, hyp) /
                self.norm_constant(observation)
            )
        return posteriors

    def compute_posterior(self, observations):
        posteriors = []
        for obs in observations:
            posteriors = self.single_posterior_update(obs)
            self.priors = posteriors
        return posteriors


if __name__ == '__main__':
    # The Cookie Problem (1)
    hypotheses_1 = ["Bowl1", "Bowl2"]
    priors_1 = [0.5, 0.5]
    observations_1 = ["chocolate", "vanilla"]
    likelihoods_1 = [[15/50, 35/50], [30/50, 20/50]]
    # Question 1:
    b = Bayes(hypotheses_1, priors_1, observations_1, likelihoods_1)
    q1 = round(b.single_posterior_update("vanilla")[hypotheses_1.index("Bowl1")], 3)
    msg = f'Question 1: {q1}'
    print(msg)
    with open('group_16.txt', 'w') as f:
        f.write(msg)
    # Question 2:
    q2 = round(b.compute_posterior(["chocolate", "vanilla"])[hypotheses_1.index("Bowl2")], 3)
    msg = f'Question 2: {q2}'
    print(msg)
    with open('group_16.txt', 'a') as f:
        f.write('\n' + msg)

    # The Archery Problem (2)
    hypotheses_2 = ['beginner', 'intermediate', 'advanced', 'expert']
    priors_2 = [0.25, 0.25, 0.25, 0.25]
    observations_2 = ['yellow', 'red', 'blue', 'black', 'white']
    likelihoods_2 = [[0.05, 0.1, 0.4, 0.25, 0.2],
                     [0.1, 0.2, 0.4, 0.2, 0.1],
                     [0.2, 0.4, 0.25, 0.1, 0.05],
                     [0.3, 0.5, 0.125, 0.05, 0.025]]
    b = Bayes(hypotheses_2, priors_2, observations_2, likelihoods_2)
    # Question 3:
    posteriors_2 = b.compute_posterior(['yellow', 'white', 'blue', 'red', 'red', 'blue'])
    q3 = round(posteriors_2[hypotheses_2.index('intermediate')], 3)
    msg = f'Question 3: {q3}'
    print(msg)
    with open('group_16.txt', 'a') as f:
        f.write('\n' + msg)
    # Question 4:
    q4 = hypotheses_2[posteriors_2.index(max(posteriors_2))]
    msg = f'Question 4: {q4}'
    print(msg)
    with open('group_16.txt', 'a') as f:
        f.write('\n' + msg)
