import matplotlib.pyplot as plt
import torch


def gp_plot(observed_pred, test_x, train_x=None, train_y=None):
    
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    
    with torch.no_grad():

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        if train_x is not None and train_y is not None:
            # Plot training data as black stars
            ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        # ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])

    return ax
