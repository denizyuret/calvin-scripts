# controller coordinates:

Here is a mapping from controller coordinates to tcp coordinates:

### door (slider): 

Notes: A and D have the door at the same location (to the left of the desk), B and C are
slightly off. All scenes have door at y=0, z=0.53. The tcp.x and door.x have reverse
directions. The range for door.x is 0.00:30.00. TODO: Check the B-C difference!

```
door in A:         range=[-0.002359:0.283729]	From calvin_table_A.urdf: base__slide
tcpx = 0.04-doorx  range=[-0.290013:0.111992]	.8*jointx = -0.10 (the center?)
tcpy = 0.00        range=[-0.083748:0.063747]	.8*jointy = 0.075
tcpz = 0.53        range=[ 0.463807:0.607193]	.8*jointz = 0.535264

door in B:         range=[-0.001999:0.306696]
tcpx = 0.23-doorx  range=[-0.182045:0.327057]	.8*jointx = 0.088
tcpy = 0.00        range=[-0.066444:0.069018]	.8*jointy = 0.075
tcpz = 0.53        range=[ 0.493990:0.660118]	.8*jointz = 0.535264

door in C:         range=[-0.002693:0.242157]
tcpx = 0.20-doorx  range=[-0.075764:0.284265]	.8*jointx = 0.06
tcpy = 0.00        range=[-0.054552:0.047173]	.8*jointy = 0.075
tcpz = 0.53        range=[ 0.495090:0.606924]	.8*jointz = 0.535264

door in D:         range=[-0.002078:0.281810]
tcpx = 0.04-doorx  range=[-0.312252:0.110304]	.8*jointx = -0.10
tcpy = 0.00        range=[-0.062834:0.041952]	.8*jointy = 0.075
tcpz = 0.53        range=[ 0.485743:0.611444]	.8*jointz = 0.535264
```

### drawer:

Notes: All scenes have drawer at z=0.36. A and C have it at x=0.10, B and D have it at
x=0.18. Drawer operates on the y axis with the door coordinate the opposite of the y
coordinate. The range of door movement is 0.00:0.22.

```
drawer in A:        range=[-0.000782:0.220825]	From calvin_table_A.urdf: base__drawer
tcpx = 0.10         range=[-0.009829:0.172383]	.8*jointx = 0.10
tcpy = -0.20-drawer range=[-0.455591:-0.188756]	.8*jointy = 0.00
tcpz = 0.36         range=[ 0.300350:0.543916]	.8*jointz = 0.32 (bottom of drawer? handle may be in the middle)

drawer in B:        range=[-0.000748:0.220765]
tcpx = 0.18         range=[ 0.091167:0.293705]	.8*jointx = 0.18
tcpy = -0.20-drawer range=[-0.476947:-0.192490]	.8*jointy = 0.00
tcpz = 0.36         range=[ 0.304156:0.523035]	.8*jointz = 0.32

drawer in C:        range=[-0.000493:0.220602]
tcpx = 0.10         range=[ 0.034033:0.160883]	.8*jointx = 0.10
tcpy = -0.20-drawer range=[-0.451620:-0.191535]	.8*jointy = 0.00
tcpz = 0.36         range=[ 0.293439:0.518048]	.8*jointz = 0.32

drawer in D:        range=[-0.000900:0.220821]
tcpx = 0.18         range=[ 0.095772:0.253271]	.8*jointx = 0.18
tcpy = -0.20-drawer range=[-0.463522:-0.193341]	.8*jointy = 0.00
tcpz = 0.36         range=[ 0.296918:0.513985]	.8*jointz = 0.32
```


### switch:

Notes: The switch moves in the Y-Z plane. Its x position is around 0.27 in A&D, around -0.32
in B&C (probably should pool their data). Its movement range is 0.00:0.09. However the y-z
coefficients are all different: either the tip of the tool does not move with the slanted
switch or it is installed at a different angle at every scene (not the case). Also the
gripper has to be under or over the switch to push it up or down, which adds an offset
because of the thickness. Maybe can do better with access to simulator code. We also need to
know which point tcp refers to exactly.

```
switch in A:           range=[-0.001835:0.088369]	From calvin_table_A.urdf: base__switch
tcpx = 0.26            range=[ 0.243142:0.306039]	.8*jointx = 0.30
tcpy = 0.0296+0.4168s  range=[-0.008334:0.084772]	.8*jointy = 0.060928
tcpz = 0.5412+0.8837s  range=[ 0.478306:0.639290]	.8*jointz = 0.494888

switch in B:           range=[-0.001319: 0.090159]
tcpx = -0.31           range=[-0.350396:-0.275247]	.8*jointx = -0.32
tcpy = 0.0186+0.4374s  range=[-0.009503: 0.087056]	.8*jointy = 0.060928
tcpz = 0.5412+0.6140s  range=[ 0.484341: 0.637629]	.8*jointz = 0.494888

switch in C:           range=[-0.002909: 0.088975]
tcpx = -0.32           range=[-0.337730:-0.294614]	.8*jointx = -0.32
tcpy = 0.0186+0.4705s  range=[-0.006038: 0.080751]	.8*jointy = 0.060928
tcpz = 0.5507+0.6574s  range=[ 0.486650: 0.638358]	.8*jointz = 0.494888

switch in D:           range=[-0.002888: 0.088266]
tcpx = 0.28            range=[ 0.250364: 0.317681]	.8*jointx = 0.30
tcpy = 0.0238+0.4091s  range=[-0.007851: 0.074466]	.8*jointy = 0.0609282
tcpz = 0.5467+0.8044s  range=[ 0.496080: 0.638987]	.8*jointz = 0.49489
```


### button:

Notes: There is some variation in the X-Y coordinates because the button is big and the tcp
is not always in the same position. However the mean should give us an idea about the center
of the button. It seems to be fixed at Y=-0.12 and A.X=-0.27 B.X=+0.28 C.X~=D.X=-0.12. The
button moves on the Z axis, its range is small [0.00-0.03]. The button coordinate has the
opposite direction from the Z coordinate. Interestingly the coefficient does not seem to be
-1.0, so the button state is not measured in meters?

```
button in A:           range=[ 0.000060: 0.032333]	From: calvin_table_A.urdf: base__button
tcpx = -0.27           range=[-0.306333:-0.163803]	.8*jointx = -0.28
tcpy = -0.11           range=[-0.159289:-0.070262]	.8*jointy = -0.10
tcpz = 0.5163-1.74b    range=[ 0.467474: 0.574068]	.8*jointz = 0.46 (probably the bottom where it touches the table)

button in B:           range=[-0.000009: 0.025924]
tcpx =  0.28           range=[ 0.152613: 0.303529]	.8*jointx = 0.28
tcpy = -0.12           range=[-0.189507:-0.069197]	.8*jointy = -0.12
tcpz = 0.5169-1.87b    range=[ 0.473718: 0.592380]	.8*jointz = 0.46

button in C:           range=[ 0.000064: 0.030551]
tcpx = -0.11           range=[-0.171218:-0.032428]	.8*jointx = -0.12
tcpy = -0.12           range=[-0.157899:-0.074686]	.8*jointy = -0.12
tcpz = 0.5148-1.70b    range=[ 0.469201: 0.590159]	.8*jointz = 0.46

button in D:           range=[ 0.000064: 0.033721]
tcpx = -0.12           range=[-0.162780:-0.041127]	.8*jointx = -0.12
tcpy = -0.12           range=[-0.155131:-0.082582]	.8*jointy = -0.12
tcpz = 0.5158-1.76b    range=[ 0.465889: 0.567390]	.8*jointz = 0.46
```
