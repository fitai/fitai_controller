import matplotlib.pyplot as plt


def create_4sig_plot(sig1, sig2, sig3, sig4, colors=['black', 'green', 'blue', 'purple'],
                     line_styles=['-', '-', '-', '--'], legend=['sig1', 'sig2', 'sig3', 'sig4'], title=''):
    plt.figure(1)
    plt.plot(sig1, color=colors[0], linestyle=line_styles[0])
    plt.plot(sig2, color=colors[1], linestyle=line_styles[1])
    plt.plot(sig3, color=colors[2], linestyle=line_styles[2])
    plt.plot(sig4, color=colors[3], linestyle=line_styles[3])
    plt.legend(legend)
    plt.title(title)


def create_3sig_plot(sig1, sig2, sig3, colors=['black', 'blue', 'green'], legend=['sig1', 'sig2', 'sig3'], title=''):
    plt.plot(sig1, color=colors[0])
    plt.plot(sig2, color=colors[1])
    plt.plot(sig3, color=colors[2])
    plt.legend(legend)
    plt.title(title)
