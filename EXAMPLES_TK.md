This file contains some common TK Console commands to manipulate VMD
Good for safekeeping!

```
  # set the unit cell side length to 10 in all frames
  pbc set {10.0 10.0 10.0} -all

  # enforce PBC
  pbc wrap -all

  # disable PBC
  pbc unwrap -all
  
  # draw a box, centered on the unitcell 
  pbc box -center unitcell
```