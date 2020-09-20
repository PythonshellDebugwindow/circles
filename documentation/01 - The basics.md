# The basics
Circles is an esoteric programming language, meaning it's a programming language that was created not in seriousness, but just for fun. Unlike more traditional languages, whose programs are text files, Circles programs are image files. This is because a program in Circles looks like a maze of nested circles, connected by parallel lines which themselves can contain circles.

# 1 Types of circles
There are two types of circles in Circles: *normal circles* and *start circles*. There is one start circle, and an unlimited amount of normal circles, in each program.

# 1.1 Start circles
![A start circle.](../images/start-circle.png?raw=true)
<br />
*A start circle.*

Each program has one start circle. This is the "entry point" of the program, or where the program starts execution. If this circle is ever reentered, the program is terminated.

# 1.2 Normal circles
![A normal circle.](../images/normal-circle.png?raw=true)
<br />
*A normal circle.*

Each program can have unboundedly many normal circles. They can hold signed integer values, which can vary over the course of a program. Normal circles are connected to each other, and to start circles, by lines (*paths*); a program can have unlimited amounts of these paths.

# 1.3 Incrementation circles
![An implementation circle.](../images/incrementation-circle.png?raw=true)
<br />
*An incrementation circle.*

Incrementation circles act just like normal circles, except that the next normal circle reached by the program will have its value incremented. If, at any time during program execution, the program is at an incrementation circle and there are no normal circles directly reachable from the incrementation circle (without going backwards, of course, as that is illegal), an error is thrown.

# 2 Types of paths

# 2.1 Normal paths
![A normal path.](../images/normal-path.png?raw=true)
<br />
*A normal path.*

Normal paths are the most basic types of paths: they are used exclusively to connect circles. Execution consists mainly of traversing these paths, moving from circle to circle. If there are ever two or more possible normal paths that can be taken from a single circle, an error will be thrown.

# 2.2 Priority paths
![A priority path.](../images/priority-path.png?raw=true)
<br />
*A priority path.*

Priority paths are almost exactly like normal paths, but with one difference: the take precedence over normal paths. For example, if a program's start circle has both a normal path and a priority path connected to it, instead of an error being thrown, the priority path will be chosen over the normal path; however, if there are ever two or more possible priority paths that can be taken from a single circle, an error will be thrown.

# 1.3 Conditional priority paths
![A conditional priority path.](../images/conditional-priority-path.png?raw=true)
<br />
*A conditional priority path.*

Conditional priority paths are the same as priority paths, except that if the value of the circle which led the program to this path is zero or less, the path will be ignored.
