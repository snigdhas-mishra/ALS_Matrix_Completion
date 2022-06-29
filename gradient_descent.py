import autograd.numpy as gnp
from autograd import grad


ALPHA_U = 0.1
ALPHA_V = 0.1


def als_gradient_descent_U_step(obs,mask,U,V,args,l2=1.0):
    global ALPHA_U
    alpha_u = float(ALPHA_U)
    def loss_U(U_i):
        preds = gnp.dot(U_i,V.T)
        loss = gnp.sum(mask *((obs - preds)**2)) / gnp.sum(mask) + l2 * gnp.mean(U_i**2)
        return loss

    loss_grad_func = grad(loss_U)
    if args.verbose:
        print('\t TRAINING:\t\tU step initial loss',loss_U(U))
    while True:
        U_current = gnp.array(U)
        objs=[loss_U(U_current)]
        for i in range(args.max_GD_steps):
            #print(obj_value)
            U_delta = loss_grad_func(U_current)
            U_current = U_current - alpha_u * U_delta
            obj_value = loss_U(U_current)
            objs.append(obj_value)
            if objs[-1] -objs[-2] >5.0:
                break
        if objs[-1] - objs[-2] > 5.0:
            if args.verbose:
                print('learning rate is too high. Dividing by half ...')
            alpha_u = alpha_u /2
            continue
        else:
            if args.verbose:
                print('\t TRAINING:\t\tU Trained with alpha', alpha_u)
                print('\t TRAINING:\t\tU step final loss', loss_U(U_current))
            break

    return U_current

def als_gradient_descent_V_step(obs,mask,U,V,args,l2=1.0):
    global ALPHA_V
    def loss_V(V_i):
        preds = gnp.dot(U,V_i.T)
        #loss = gnp.sum(mask * gnp.sum((obs - preds)**2) / gnp.sum(mask)) + l2 * gnp.mean(V_i**2)
        loss = gnp.sum(mask *((obs - preds)**2)) / gnp.sum(mask) + l2 * gnp.mean(V_i**2)
        return loss

    loss_grad_func = grad(loss_V)
    if args.verbose:
        print('\t TRAINING:\t\tV step initial loss',loss_V(V))
    while True:
        V_current = gnp.array(V)
        objs=[loss_V(V_current)]
        for i in range(args.max_GD_steps):
            #print(obj_value)
            V_delta = loss_grad_func(V_current)
            V_current = V_current - ALPHA_V * V_delta
            obj_value = loss_V(V_current)
            objs.append(obj_value)
            if objs[-1] -objs[-2] >5.0:
                break
        if objs[-1] - objs[-2] > 5.0:
            if args.verbose:
                print('learning rate is too high. Dividing by half ...')
            ALPHA_V = ALPHA_V /2
            continue
        else:
            if args.verbose:
                print('\t TRAINING:\t\tV Trained with alpha', ALPHA_V)
                print('\t TRAINING:\t\tV step final loss', loss_V(V_current))
            break
    return V_current







