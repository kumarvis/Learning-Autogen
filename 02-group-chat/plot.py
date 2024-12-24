import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle

# Initialize the plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis("off")

# Function to draw a block with text
def draw_block(ax, x, y, text, width=2, height=1):
    rect = Rectangle((x, y), width, height, edgecolor="black", facecolor="lightblue", lw=2)
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10, wrap=True)
    return rect

# Function to draw an arrow
def draw_arrow(ax, start, end):
    arrow = FancyArrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                       width=0.05, head_width=0.2, length_includes_head=True, color="black")
    ax.add_patch(arrow)

# Draw blocks
blocks = {}
blocks["User_Proxy"] = draw_block(ax, 0, 6, "User_Proxy")
blocks["Manager"] = draw_block(ax, 4, 6, "Manager")
blocks["Planner"] = draw_block(ax, 4, 4, "Planner")
blocks["Engineer"] = draw_block(ax, 2, 2, "Engineer")
blocks["Scientist"] = draw_block(ax, 6, 2, "Scientist")
blocks["Critic"] = draw_block(ax, 4, 0, "Critic")

# Draw arrows for communication flow
draw_arrow(ax, (1, 6.5), (4, 6.5))  # User_Proxy -> Manager
draw_arrow(ax, (4.5, 6), (4.5, 4))  # Manager -> Planner
draw_arrow(ax, (4, 4), (2, 2.5))    # Planner -> Engineer
draw_arrow(ax, (4.5, 4), (6, 2.5))  # Planner -> Scientist
draw_arrow(ax, (4.5, 4), (4.5, 0))  # Planner -> Critic
draw_arrow(ax, (2.5, 2), (4, 4.5))  # Engineer -> Planner
draw_arrow(ax, (6.5, 2), (4, 4.5))  # Scientist -> Planner
draw_arrow(ax, (4.5, 0), (4, 4.5))  # Critic -> Planner
draw_arrow(ax, (4.5, 6), (1, 6.5))  # Manager -> User_Proxy

# Add annotations for clarity
ax.text(4, 7, "Task Initiation", fontsize=12, ha="center")
ax.text(4, 3, "Plan Refinement", fontsize=12, ha="center", va="bottom")
ax.text(4, -1, "Feedback Loop", fontsize=12, ha="center", va="top")

# Display the chart
#plt.show()
plt.plot()
plt.savefig('message-flow.png')
plt.close()
