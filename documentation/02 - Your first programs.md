**<-** [**Previous**](./01%20-%20The%20basics.md)

# Your first programs

## 1 A basic program
![A basic program](../images/program-1.png?raw=true)

The above program is very simple indeed: it starts in the start circle (top left), crosses the normal path, and ends up in the normal circle (bottom right); since there is nowhere to go from the normal circle without traversing the path which led to the circle (which is illegal), the program halts. This program is basically just a fancy nop, but demonstrates some basic features of the language, such as circles and normal paths.

## 2 Another basic program
![Another basic program](../images/program-2.png?raw=true)

The program shown here appears complicated than the previous one, yet is in reality still fairly simple. It starts in the start circle (top left), and then moves southeast to the northwesternmost normal circle. From there, it moves east instead of southeast, due to the priority path, then moves southwest, then northwest, and then sees the priority path once more and goes down it again, and thus the cycle continues for all of eternity (or until you kill the program). This program is an infinite loop, and demonstrates priority paths.

## 3 A slightly more advanced program
![A slightly more advanced program](../images/program-3.png?raw=true)

This program demonstrates the use of priority paths, as well as what happens when paths are ambiguous. The program starts in the start circle (top left), then goes down instead of right because the path going down is a priority path, while the path going to the right is just a normal path. The normal circle it arrives at has two possible paths (going right and down), both normal this time, so an error is thrown due to the ambiguity created.

## 4 A program using incrementation circles
![A program using incrementation circles](../images/program-4.png?raw=true)

The program pictured here utilises an incrementation circle. It starts in the start circle, then follows two normal paths (to the right and down-right (down and to the right) respectively), then goes down-left due to the priority path, then right. It then hits an incrementation circle. After following the next normal path up-left, it hits a normal circle and increments its value. It then takes the priority path again to the down-left, and this cycle of incrementation continues forever. If left to run for an infinite time, the value of the normal circle at the top of the triangle at the bottom of the program will be infinite.

## 5 A while loop
![A while loop](../images/program-5.png?raw=true)

The program depicted above implements a while loop. First, the program increments the circle on the left side of the longest path; let's call that circle C. Then, since C's value is three, which is, of course, greater than zero, the conditional priority path is taken and C's value is decremented. This repeats twice; after the third time, C's value is zero, and so the conditional priority path is ignored and the priority path is taken; since there no other paths leading out of the circle which the priority path leads to, the program halts. The southernmost two circles can be thought of as the "body" of the while loop, and the conditional priority path as the condition (although this means that the condition must always be `x > 0` where `x` is C's value).

## 6 A program that outputs the number one
![A program that outputs the number one](../images/program-6.png?raw=true)

The program shown above outputs the number one. It first increments the value of the circle in the exact center of the program, which we'll call C here for ease of reference. The program then take the conditional priority path to the down-right into the output circle, writing C's value, which is currently 1, to standard output. C's value is then decremented, and the priority path is taken to the leftmost normal circle; C's value is now zero, so the conditional priority path is ignored. Since the leftmost normal circle is a dead end, having no other paths leading away from it, the program terminates.

## 7 A [truth-machine](https://www.esolangs.org/wiki/Truth-machine)
![A truth-machine](../images/program-7.png?raw=true)

The program shown above is a truth-machine: it halts if you input the number zero, or loops forever if you input the number one; all other input is undefined behaviour and should not be used. The program starts in the start circle, reads an integer from standard input, and stores that integer in the top-central circle, which we'll call C here for the sake of brevity. If C's value is non-zero (e.g., if the user input the number one), then the program will take the conditional priority path to the down-right, then go left, then up-right, and then repeat that loop forever; if C's value is zero (e.g., if the user input the number zero), then the program will go to the right and halt.

## 8 Writing your own programs
I used [Scratch](https://scratch.mit.edu/) to make the programs in this file. The specific project I used was [this one](https://scratch.mit.edu/projects/429742899/); hit `See inside` and drag the sprites around to create programs, and take a partial screenshot of the canvas to save it.

> Hi, Cally here. You should use [the editor I made with Godot](https://github.com/photon-niko/circles-editor) that is up-to-date, easier to use, and just looks nicer :3

**<-** [**Previous**](./01%20-%20The%20basics.md)
