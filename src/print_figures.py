import matplotlib.pyplot as plt

def draw_bar(labels, label_counts, y_axis_name="", x_axis_name="", label_color_dict= {}, title="", legend_title="", filename=""):

  labels = list(label_color_dict.keys())
  handles = handles = [plt.Rectangle((0,0),1,1, color=label_color_dict[label]) for label in labels]
  colors = list(label_color_dict.values())
  fig, ax = plt.subplots() 

  ax.bar(labels, label_counts, label=labels, color=colors)

  ax.set_ylabel(ylabel=y_axis_name)
  ax.set_xlabel(xlabel=x_axis_name)

  ax.set_title(title)
  
  ax.legend(handles, labels, title=legend_title)
  fig.savefig(filename)
  fig.show()
