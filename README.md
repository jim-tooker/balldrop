
# Ball Drop Simulator
(Package `balldrop`)

This package simulates the dropping of balls under gravity and different environmental conditions. The simulation can handle multiple balls, track their velocity, acceleration, position over time, and display these values graphically..

Here are the runtime options: `python ball_drop_sim.py -h`:
```
usage: ball_drop_sim.py [-h] [--test] [--no_gui]

Ball Drop Simulator

options:
  -h, --help  show this help message and exit
  --test      Run with a pre-defined test case
  --no_gui    Run without GUI
```

## Modules:
* `ball_drop_sim`: The main module that runs the system. It manages the entire simulation process, including initialization, running the simulation, and displaying the results.
* `ball`: Contains the `Ball` class which represents the balls in the system.
* `ball_spec`: Contains classes that define the specifications for a ball.
* `environment`: Contains the classes that define the environment in which the ball drop simulation occurs.

## Documentation
For detailed API documentation, see:
[Ball Drop Simulator API Documentation](https://jim-tooker.github.io/balldrop/docs/ball_drop_sim.html)

## To Use
1. Clone this repository.
2. Run the setup script: `setup.sh`.  *(This will install needed dependencies)*
3. Run `python ball_drop_sim.py`.  *(With or without command line options)*

## Sample Screenshot
<img width="1363" alt="ball-drop" src="https://github.com/user-attachments/assets/32193b42-99d9-4fd2-9420-efbc5bb87f4f">


## Sample Video of Execution
https://github.com/user-attachments/assets/e152d68c-849b-4805-b922-130f187a0808


