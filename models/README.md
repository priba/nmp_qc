#Available MPNN models

Some of the models available in the literature have been implemented as Message, Update and Readout functions.

## MpnnDuvenaud

This class implements the whole Duvenaud et al. model following the functions proposed by Gilmer et al. as Message, Update and Readout.

``` 
Parameters
----------
    d : int list.
        Possible degrees for the input graph.
    in_n : int list
        Sizes for the node and edge features.
    out_update : int list
        Output sizes for the different Update funtion.
    hidden_state_readout : int
        Input size for the neural net used inside the readout function.
    l_target : int
        Size of the output.
    type : str (Optional)
        Classification | [Regression (default)]. If classification, LogSoftmax layer is applied to the output vector.
```

Definition:

``` 
    model = MpnnDuvenaud(d, in_n, out_update, hidden_state_readout, l_target')
```

## MpnnGGNN


This class implements the whole Li et al. model following the functions proposed by Gilmer et al. as Message, Update and Readout.

``` 
Parameters
----------
    e : int list.
        Possible edge labels for the input graph.
    hidden_state_size : int
        Size of the hidden states (the input will be padded with 0's to this size).
    message_size : int
        Message function output vector size.
    n_layers : int
        Number of iterations Message+Update (weight tying).
    l_target : int
        Size of the output.
    type : str (Optional)
        Classification | [Regression (default)]. If classification, LogSoftmax layer is applied to the output vector.
```

Definition:

``` 
    model = MpnnGGNN(e, in_n, hidden_state_size, message_size, n_layers, l_target)
```
 
## IntNet

This class implements the whole Battaglia et al. model following the functions proposed by Gilmer et al. as Message, Update and Readout.

``` 
Parameters
----------
    in_n : int list
        Sizes for the node and edge features.
    out_message : int list
        Output sizes for the different Message functions.
    out_update : int list
        Output sizes for the different Update functions.
    l_target : int
        Size of the output.
    type : str (Optional)
        Classification | [Regression (default)]. If classification, LogSoftmax layer is applied to the output vector.
```

Definition:

``` 
    model = MpnnIntNet(in_n, out_message, out_update, l_target):
```

## MPNN as proposed by Gilmer et al.

This class implements the whole Gilmer et al. model following the functions Message, Update and Readout.

In progress..

* [x] Edge Network
* [ ] Virtual Graph Elements
* [ ] set2set Readout function
* [ ] Multiple Towers

``` 
Parameters
----------
    in_n : int list
            Sizes for the node and edge features.
    hidden_state_size : int
        Size of the hidden states (the input will be padded with 0's to this size).
    message_size : int
        Message function output vector size.
    n_layers : int
        Number of iterations Message+Update (weight tying).
    l_target : int
        Size of the output.
    type : str (Optional)
        Classification | [Regression (default)]. If classification, LogSoftmax layer is applied to the output vector.
``` 
Definition:

``` 
    model = MPNN(in_n, hidden_state_size, message_size, n_layers, l_target, type='regression'):
``` 
