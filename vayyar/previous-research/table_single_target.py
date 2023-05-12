import matplotlib.pyplot as plt
import numpy as np


# Define Table Figure
def table_init():
    title_text = "Target's Posture"
    fig_background_color = 'skyblue'
    fig_border = 'steelblue'
    data = [
        [        'Posture'],
        ['Target', '']
    ]
    # Pop the headers from the data array
    column_headers = data.pop(0)
    row_headers = [x.pop(0) for x in data]
    # Table data needs to be non-numeric text. Format the data while I'm at it.
    cell_text = []
    for row in data:
        cell_text.append([f'{x}' for x in row])
    # Get some lists of color specs for row and column headers
    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
    # Create the figure.
    plt.figure(linewidth=2,
               edgecolor=fig_border,
               facecolor=fig_background_color,
               tight_layout={'pad': 1},
               figsize=(4, 3)
               )
    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          cellLoc='center',
                          rowLabels=row_headers,
                          rowColours=rcolors,
                          rowLoc='right',
                          colColours=ccolors,
                          colLabels=column_headers,
                          loc='center')
    # Make the rows taller (i.e., make cell y scale larger).
    the_table.scale(1, 1.5)
    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Hide axes border
    plt.box(on=None)
    # Add title
    plt.suptitle(title_text)
    # Force the figure to update, so backends center objects correctly within the figure.
    # Without plt.draw() here, the title will center on the axes and not the figure.
    plt.draw()
    # Create image. plt.savefig ignores figure edge and face colors, so map them.
    fig = plt.gcf()

    return the_table, fig
