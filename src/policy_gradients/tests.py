from torch_utils import *
import torch as ch

if __name__ == '__main__':
    def func(x):
        f1 = x[0] * x[0] * x[1]
        f2 = x[2]/(x[1] * x[0])
        f3 = x[2]
        f4 = x[0]

        return ch.stack([f1, f2, f3, f4])

    def J(x):
        return ch.tensor([
            [2 * x[0] * x[1], x[0]**2, 0],
            [-x[2] / (x[1] * x[0]**2), -x[2] / (x[0] * x[1]**2), (x[0] * x[1])**(-1)],
            [0, 0, 1],
            [1, 0, 0]
        ])

    ch.manual_seed(0)
    print('Running tests!')

    def test_fisher_vector_prod(x):
        fisher_product = 0
        for state_action_dist in pds[selected]:
            diag_M = state_action_dist.pow(-1)
            Jx = jvp(state_action_dist, net.parameters(), x)
            MJx = diag_M * Jx
            JTMJx = vjp(state_action_dist, net.parameters(), MJx, False)
            fisher_product += JTMJx
        alt = alt_fisher_product(x)
        res = fisher_product / ch.tensor(num_samples).float() + x*params.DAMPING
        print(alt, res)
        print("Correlation", (alt*res).sum()/ch.sqrt((alt**2).sum()  * (res**2).sum()))
        return res

    def test_it(t, e):
        return (t - e)/((t + e)/2)

    for i in range(5):
        x = ch.tensor(ch.rand(3), requires_grad=True)
        v = ch.rand(4)
        u = ch.rand(3)

        jacobian = J(x)
        Ju = jacobian @ u
        JTv = jacobian.t() @ v

        est_Ju = jvp(func(x),[x],u) 
        est_JTv = vjp(func(x),[x],v)

        print('Ju:', test_it(Ju, est_Ju))
        print('JTv:', test_it(JTv, est_JTv))
        print('-' * 80)
