# microRefactor
A mini-app to practice refactoring a simple weather microphysics code to run efficiently on GPUs

Author:  
**Matt Norman**  
Oak Ridge National Laboratory  
https://mrnorman.github.io

## Refactoring is more than just "porting"

We hear the word "porting" thrown around a lot, but it's not a great word for what we really have to do to codes to get them running *efficiently*.

### Why "one giant kernel" typically doesn't work
Sure, you could slap a directive around a giant section of code and put callsite directives for subroutines and functions. Sure, in the absence of compiler errors, this will get your code to run on the GPU. But nine times out of ten, you'll find the kernel runs about the same or slower than the CPU code did. There are two main reasons for this:

1. **Registers**: The more lines of code you have in a kernel, the more easily it runs out of register space on a GPU. Once you beging to overflow the registers (or even use too many of them), your performance will typically suffer significantly. The only solution is to break the kernel up into multiple kernels, each with fewer lines of code.
2. **Function calls**: True function calls on the GPU are very expensive. You need a *lot* of work to amortize the cost of a GPU function call. Sometimes the compiler can inline function calls, but when it cannot, you'll find the performance will suffer.

Most weather and climate codes codes have an outer loop over horizontal indices ("columns") that surround a lot of calls to different physics routines, and each physics routine operates only on one column at a time. Also, often times, there is some mild CPU-level OpenMP parallelism on the outer horizontal loop as shown below:

```fortran
real, dimension(nz) :: theta, u, v, w, ql, qi, qv, rho
!$omp parallel do private(theta,u,v,w,ql,qi,qv,rho)
do j = 1 , ny
  do i = 1 , nx
    do k = 1 , nz
      theta(k) = theta_glob(i,j,k)
      ! Load column for other variables ...
    enddo
    call sub_grid_scale_turbulence(theta,u,v,w,ql,qi,qv,rho)
    call microphysics(theta,w,ql,qi,qv,rho)
    call radiation(theta,ql,qi,qv,rho)
    call boundary_layer(theta,u,v,w,ql,qi,qv,rho)
    do k = 1 , nz
      theta_glob(i,j,k) = theta(k)
      ! Store column for other variables ...
    enddo
    ...
  enddo
enddo
```

Suppose you want to thread only over the horizontal loops. If you make one giant kernel on the outer horizontal loops, you'll end up with hundreds of thousands of lines of code in the kernel, which will never run efficiently on a GPU. It simply has too many lines of code. Also, you'll end up with a lot of function calls. The situation is more complicated than the above code suggests because you're typically calling routines 8-10 levels deep. Most Fortran compilers will either fail to inline this or will take hours to compile if they attempt to. If your code is in C++, you'll find the compiler inlines much more often and more quickly.

### Fission the outer loop

The only way to address this problem is to break up the outer loop into multiple loops, i.e.,


```fortran
do j = 1 , ny
  do i = 1 , nx
    call sub_grid_scale_turbulence(...)
  enddo
enddo
do j = 1 , ny
  do i = 1 , nx
    call microphysics(...)
  enddo
enddo
do j = 1 , ny
  do i = 1 , nx
    call radiation(...)
  enddo
enddo
do j = 1 , ny
  do i = 1 , nx
    call boundary_layer(...)
  enddo
enddo
```

The main problem with this is that most of the data shared among the physical routines only has a vertical dimension, `nz`, because you typically perform the outer loops sequentially. If you fission the loop between physics calls, you now have to "promote" the variables shared between physics calls to include the horizontal dimensions as well. Otherwise, you'll just write over the same column before you get to the next horizontal loop, and the data will no longer be valid.

### Promoting variables

In the example above, all of the variables need to be promoted to add the `nx` and `ny` dimensions as well. However, in more realistic codes, it is not so obvious. Often times, Fortran codes will use module-level data that's only dimensioned over `nz`, and those will have to be promoted as well so it's valid between different physics calls.

The real difficulty of this refactoring effort is locating **all** instances of the variable you're promoting and changing it to include horizontal indices. Also, often times, this requires passing the horizontal indices into subroutines so that they are available.

### Pushing looping down the callstack

The next step after fissioning the loops and promoting the variables is to push the looping down the callstack, which will change our code to look like the following in the end:

```fortran
real, dimension(nx,ny,nz) :: theta, u, v, w, ql, qi, qv, rho

call sub_grid_scale_turbulence(theta,u,v,w,ql,qi,qv,rho)
call microphysics(theta,w,ql,qi,qv,rho)
call radiation(theta,ql,qi,qv,rho)
call boundary_layer(theta,u,v,w,ql,qi,qv,rho)
```

In this code, all looping is now inside the subroutines. 

### Exposing the vertical loop to parallelism

For climate codes, which must run at 2,000 times realtime, you cannot afford to loop only over the horizontal indices. This is because there are only 100s of horizontal threads available per GPU when the code is significantly strong scaled so that it runs faster. Increasingly, weather and medium range codes are finding themselves in this boat as well. In these cases, threading over the vertical dimension as well will expose 50-100x more parallelism, which will use GPUs much more effectively. 

However, in these cases, you'll often uncover "race conditions." They are by far the most common in the vertical dimension. Race conditions are situations where two parallel threads might read and write to the same location at the same time (these need "atomic" or "reduction" instructions) or when the threads have to execute in a certain order for correctness (these are called "prefix sums" or "scans").

You can identify the need for atomics or reductions when a variable on the left of an equals sign has fewer dimensions than you have parallel loops.

You can identify a prefix sum when you have expressions like: `a(i) = a(i-1) + ...`. In these cases, the loop order matters.

#### Atomics

An example of needing these is below:

```fortran
!$acc parallel loop collapse(3)
do k = 1 , nz
  do j = 1 , ny
    do i = 1 , nx
      !$acc atomic
      theta_col_sum(k) = theta_col_sum(k) + theta(i,j,k)
    enddo
  enddo
enddo
```

The danger here is reading a stale value. Suppose that threads `i=1,j=1` and `i=2,j=1` are executed at **exactly** the same time. They will both read a given value for the current sum, say, `0`. Then they each will add their values and then write the sum. Suppose the value at `theta(1,1,1) = 300` and `theta(1,1,2)=301`. The appropriate sum is `601` for those two threads. But if they each read an existing value of `0`, and they each write to the location in memory at the same time, then you will either get `300` or `301` as the sum of the two, both of which are wrong.

The "atomic" directive keeps this from happening. It only allows a given thread to write its value to a memory location if the the value at that location hasn't changed since it was originally read in.

#### Reductions

Reductions are the other way to handle this situation. Suppose we have calculated the maximum stable time step over all cells, but we want the minumum among these for the overall model's time step. This would require a reduction clause:

```fortran
!$acc parallel loop reduction(min,dt)
do k = 1 , nz
  do j = 1 , ny
    do i = 1 , nx
      dt = min(dt , dt3d(i,j,k))
    enddo
  enddo
enddo
```

#### Prefix sums / Scans

There are times when the order of the loop matters, and often times, this means for practical purposes that that particular loop must be serialized. For instance, suppose we wanted the hydrostatic pressure from density via an integral. In this case, we need to compute what we can in parallel first. Then, we need to 

```fortran
!$acc parallel loop collapse(3)
do k = 1 , nz
  do j = 1 , ny
    do i = 1 , nx
      tmp = 0
      do ii = 1 , ngll
        tmp = tmp - density(i,j,k,ii) * gravity * quad_weight(ii)
      enddo
      hydro_rhs(i,j,k) = tmp
    enddo
  enddo
enddo
!$acc parallel loop collapse(2)
do j = 1 , ny
  do i = 1 , nx
    do k = 1 , nz
      hydro_pressure(k) = hydro_pressure(k-1) + hydro_rhs(i,j,k)
    enddo
  enddo
enddo
!$acc parallel loop collapse(3)
do k = 1 , nz
  do j = 1 , ny
    do i = 1 , nx
      ! Carry on, using hydrostatic pressure now that it's computed
    enddo
  enddo
enddo
```

## Further reading

If you want more information on this, please see the following article:

[A Practical Introduction to gpu Refactoring in Fortan with Directives for Climate](https://github.com/mrnorman/miniWeather/wiki/A-Practical-Introduction-to-GPU-Refactoring-in-Fortran-with-Directives-for-Climate)

## Did you get all that?

It was kind of a blur wasn't it? The exmaples weren't that clear. Hopefully microRefactor will make this more clear with hands on experience. The example used in here is greatly simplified, but you also probably don't want to learn from a 20K line of code example either. I hope you find this repo helpful.

