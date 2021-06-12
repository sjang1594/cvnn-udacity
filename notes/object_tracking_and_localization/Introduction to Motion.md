# **Introduction to Optical Flow**

Optical flow is used in many tracking and motion analysis applications. It works by assuming two things about image frames. 

1. The pixel intensities of an object do not change between consecutive frames.
2. The neighboring pixels have similar motion. 

It then looks at interesting points such as corners or particularly bright pixels, and tracks them from one frame to the next. Tracking a point or set of points provides information about how fast that point or object is moving and in what direction. This data also allows you to predict where an object will move next.  (`Tracking a point provides information about the **spped** of movement and data that can be used to **predict the future location** of the point`). 

---

## **How does optical flow works ?**

Assume that we have to image frames from a video, and for one point on object in image one, and we want to find out where it is in image two. Once we do, we can calculate a motion vector that describes the velocity of this point from the first frame to the next. 

---

## **Brightness Constancy Assumption**

Optical flow assumes that points in one image frame have the same intensity pixel value as those same points in the next image frame. The optical flow assumes that the color of a surface will stay the same over time. 

I<sub>1</sub>(x,y) = I<sub>2</sub>(x+u, y+v)

We need another information such as time to represent that the which points comes first. To relate image frames in space and time, we can think about these image frames in another way. The first image is just the 2D pattern of itensity that happens at time t, and the second image is the intensity pattern that happens at time t+1. In this way, we can think of a series of image frames I as a 3D volume of images with x and y coordinates, pixel values at each point, and a depth dimension of time. 

I<sub>1</sub>(x,y, t) = I<sub>2</sub>(x+u, y+v, t+1)

The equation above is known as the brightness constancy assumption. This function can be broken down into a Taylor series expension, which represents this intensity function as a summation terms. 

### Second Assumption/Constraint

You’ll note that the brightness constancy assumption gives us *one* equation swith *two* unknowns (u and v), and so we also have to have another constraint; another equation or assumption that we can use to solve this problem.

Recall that in addition to assuming brightness constancy, we also assume that neighboring pixels have similar motion. Mathematically this means that pixels in a local patch have *very* similar motion vectors. For example, think of a moving person, if you choose to track a collection of points on that person’s face, all of those points should be moving at roughly the same speed. Your nose can’t be moving the opposite way of your chin.

This means that I shouldn’t get big changes in the flow vectors (u, v), and optical flow uses this idea of motion smoothness to estimate u and v for any point.

---

## **Where is Optical Flow Used**?

So, what does it look like when you apply optical flow not just to a point, but a set of points in video? 

The goal of optical flow is, for each image frame, to compute approximate motion vectors based on how the image intensity, the patterns of dark and light pixels. have changed over time. 

The first step is to find matching feature points in between two images using a hog or corner detection that looks for matching patterns of intensity.  Then, the optical flow calculates a motion vector. The u, v for each key point in the first image frame that points to where that key point can be found in the next image. The optical flow shows a set of points as direction or flow, and can be calculated frame by frame.  s