"""
This is based on the slice sampler from the old spearmint repo.
I've added the possibility to set constraints so that
the sampler doesn't explore areas that we know are badly defined.
"""
import numpy as np
import numpy.random as npr


def slice_sample(init_x, logprob, sigma=1.0, bounds=None,
                 step_out=True, max_steps_out=1000, verbose=False):
    """
    Return a sample via slice sampling.

    This function was adapted from the old Spearmint repository
    https://github.com/JasperSnoek/spearmint

    If "bounds" are provided, then the slice sampler will not expand past
    those limits.

    This function defines a direction randomly and then performs
    slice sampling along that line.

    :init_x (ndarray): column vector with the starting location

    :logprob (func): function handle that returns the value of the
                     (unnormalised) probablity density at a given x

    :sigma (float): initial size of the slice sampler's region

    :bounds (ndarray): Dx2 array with the lower and upper limits for x
                       If not provided, then the limits are [-np.inf, np.inf]

    :max_steps_out (int): maximum number of times the slice sampler's
                          region will be expanded to contain all the relevant
                          probability mass
    """
    def direction_slice(direction, init_x, z_lim=None):
        def dir_logprob(z):
            return logprob(direction * z + init_x)

        upper = sigma * npr.rand()
        lower = upper - sigma
        if z_lim is not None:
            lower = max(lower, z_lim[0])
            upper = min(upper, z_lim[1])

        # initialise at a value slightly lower than logprob(init_x)
        llh_s = np.log(npr.rand()) + dir_logprob(0.0)

        l_steps_out = 0
        u_steps_out = 0
        if step_out:
            while dir_logprob(lower) > llh_s and l_steps_out < max_steps_out:
                l_steps_out += 1
                lower -= sigma
                if z_lim is not None:
                    if lower <= z_lim[0]:
                        lower = z_lim[0]
                        break
            while dir_logprob(upper) > llh_s and u_steps_out < max_steps_out:
                u_steps_out += 1
                upper += sigma
                if z_lim is not None:
                    if upper >= z_lim[1]:
                        upper = z_lim[1]
                        break

        steps_in = 0
        while True:
            steps_in += 1
            new_z = (upper - lower) * npr.rand() + lower
            new_llh = dir_logprob(new_z)
            if np.isnan(new_llh):
                print(new_z, direction * new_z + init_x,
                      new_llh, llh_s, init_x, logprob(init_x))
                raise Exception("Slice sampler got a NaN")
            if new_llh > llh_s:
                break
            elif new_z < 0:
                lower = new_z
                if z_lim is not None:
                    if lower <= z_lim[0]:
                        lower = z_lim[0]
            elif new_z > 0:
                upper = new_z
                if z_lim is not None:
                    if upper >= z_lim[1]:
                        upper = z_lim[1]
            else:
                raise Exception("Slice sampler shrank to zero!")

        if verbose:
            print("Steps Out:", l_steps_out,
                  u_steps_out, " Steps In:", steps_in)

        return new_z * direction + init_x

    if not init_x.shape:
        init_x = np.array([init_x])

    dims = init_x.shape[0]
    direction = npr.randn(dims)
    direction = direction / np.sqrt(np.sum(direction**2))

    if bounds is None:
        z_lim = None
    else:
        # Project the bounds onto the chosen direction
        # z-values that satisfy the equality constraint for each limit
        z_lim = np.sort((bounds - init_x[:, None]) / direction[:, None])
        # print(z_lims)
        # pick out the values that are closest to the origin,
        # as these will satisfy all constraints
        z_lim = z_lim[np.argmin(np.abs(z_lim), 0), np.arange(2)]

    return direction_slice(direction, init_x, z_lim=z_lim)
