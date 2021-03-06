??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-0-gc256c071bb28??
?
%source_separation_model/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%source_separation_model/conv2d/kernel
?
9source_separation_model/conv2d/kernel/Read/ReadVariableOpReadVariableOp%source_separation_model/conv2d/kernel*'
_output_shapes
:?*
dtype0
?
#source_separation_model/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#source_separation_model/conv2d/bias
?
7source_separation_model/conv2d/bias/Read/ReadVariableOpReadVariableOp#source_separation_model/conv2d/bias*
_output_shapes
:*
dtype0
?
'source_separation_model/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'source_separation_model/conv2d_1/kernel
?
;source_separation_model/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp'source_separation_model/conv2d_1/kernel*&
_output_shapes
:*
dtype0
?
%source_separation_model/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%source_separation_model/conv2d_1/bias
?
9source_separation_model/conv2d_1/bias/Read/ReadVariableOpReadVariableOp%source_separation_model/conv2d_1/bias*
_output_shapes
:*
dtype0
?
$source_separation_model/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*5
shared_name&$source_separation_model/dense/kernel
?
8source_separation_model/dense/kernel/Read/ReadVariableOpReadVariableOp$source_separation_model/dense/kernel*
_output_shapes
:	?*
dtype0
?
"source_separation_model/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"source_separation_model/dense/bias
?
6source_separation_model/dense/bias/Read/ReadVariableOpReadVariableOp"source_separation_model/dense/bias*
_output_shapes	
:?*
dtype0
?
&source_separation_model/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&source_separation_model/dense_1/kernel
?
:source_separation_model/dense_1/kernel/Read/ReadVariableOpReadVariableOp&source_separation_model/dense_1/kernel*
_output_shapes
:	?*
dtype0
?
$source_separation_model/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$source_separation_model/dense_1/bias
?
8source_separation_model/dense_1/bias/Read/ReadVariableOpReadVariableOp$source_separation_model/dense_1/bias*
_output_shapes
:*
dtype0
?
&source_separation_model/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*7
shared_name(&source_separation_model/dense_2/kernel
?
:source_separation_model/dense_2/kernel/Read/ReadVariableOpReadVariableOp&source_separation_model/dense_2/kernel*
_output_shapes
:	?*
dtype0
?
$source_separation_model/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$source_separation_model/dense_2/bias
?
8source_separation_model/dense_2/bias/Read/ReadVariableOpReadVariableOp$source_separation_model/dense_2/bias*
_output_shapes
:*
dtype0
?
/source_separation_model/conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/source_separation_model/conv2d_transpose/kernel
?
Csource_separation_model/conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOp/source_separation_model/conv2d_transpose/kernel*&
_output_shapes
:*
dtype0
?
-source_separation_model/conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-source_separation_model/conv2d_transpose/bias
?
Asource_separation_model/conv2d_transpose/bias/Read/ReadVariableOpReadVariableOp-source_separation_model/conv2d_transpose/bias*
_output_shapes
:*
dtype0
?
1source_separation_model/conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31source_separation_model/conv2d_transpose_1/kernel
?
Esource_separation_model/conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOp1source_separation_model/conv2d_transpose_1/kernel*'
_output_shapes
:?*
dtype0
?
/source_separation_model/conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/source_separation_model/conv2d_transpose_1/bias
?
Csource_separation_model/conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOp/source_separation_model/conv2d_transpose_1/bias*
_output_shapes
:*
dtype0
?
1source_separation_model/conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31source_separation_model/conv2d_transpose_2/kernel
?
Esource_separation_model/conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOp1source_separation_model/conv2d_transpose_2/kernel*&
_output_shapes
:*
dtype0
?
/source_separation_model/conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/source_separation_model/conv2d_transpose_2/bias
?
Csource_separation_model/conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOp/source_separation_model/conv2d_transpose_2/bias*
_output_shapes
:*
dtype0
?
1source_separation_model/conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31source_separation_model/conv2d_transpose_3/kernel
?
Esource_separation_model/conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOp1source_separation_model/conv2d_transpose_3/kernel*'
_output_shapes
:?*
dtype0
?
/source_separation_model/conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/source_separation_model/conv2d_transpose_3/bias
?
Csource_separation_model/conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOp/source_separation_model/conv2d_transpose_3/bias*
_output_shapes
:*
dtype0
?
&source_separation_model/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*7
shared_name(&source_separation_model/dense_3/kernel
?
:source_separation_model/dense_3/kernel/Read/ReadVariableOpReadVariableOp&source_separation_model/dense_3/kernel*
_output_shapes

:<*
dtype0
?
$source_separation_model/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$source_separation_model/dense_3/bias
?
8source_separation_model/dense_3/bias/Read/ReadVariableOpReadVariableOp$source_separation_model/dense_3/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?3
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?2
value?2B?2 B?2
?
	conv1
	conv2
d1
dClass1
dClass2
conv3Class1
conv4Class1
conv3Class2
	conv4Class2


concat
out
	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
h

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
h

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
h

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
h

5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
h

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
h

Akernel
Bbias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
R
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
h

Kkernel
Lbias
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
?
0
1
2
3
4
5
#6
$7
)8
*9
/10
011
512
613
;14
<15
A16
B17
K18
L19
?
0
1
2
3
4
5
#6
$7
)8
*9
/10
011
512
613
;14
<15
A16
B17
K18
L19
 
?
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
 
b`
VARIABLE_VALUE%source_separation_model/conv2d/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE#source_separation_model/conv2d/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
regularization_losses
db
VARIABLE_VALUE'source_separation_model/conv2d_1/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE%source_separation_model/conv2d_1/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
^\
VARIABLE_VALUE$source_separation_model/dense/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUE"source_separation_model/dense/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
 trainable_variables
!regularization_losses
ec
VARIABLE_VALUE&source_separation_model/dense_1/kernel)dClass1/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$source_separation_model/dense_1/bias'dClass1/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1

#0
$1
 
?
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
%	variables
&trainable_variables
'regularization_losses
ec
VARIABLE_VALUE&source_separation_model/dense_2/kernel)dClass2/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$source_separation_model/dense_2/bias'dClass2/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1

)0
*1
 
?
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
+	variables
,trainable_variables
-regularization_losses
rp
VARIABLE_VALUE/source_separation_model/conv2d_transpose/kernel-conv3Class1/kernel/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE-source_separation_model/conv2d_transpose/bias+conv3Class1/bias/.ATTRIBUTES/VARIABLE_VALUE

/0
01

/0
01
 
?
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
1	variables
2trainable_variables
3regularization_losses
tr
VARIABLE_VALUE1source_separation_model/conv2d_transpose_1/kernel-conv4Class1/kernel/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE/source_separation_model/conv2d_transpose_1/bias+conv4Class1/bias/.ATTRIBUTES/VARIABLE_VALUE

50
61

50
61
 
?
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
7	variables
8trainable_variables
9regularization_losses
tr
VARIABLE_VALUE1source_separation_model/conv2d_transpose_2/kernel-conv3Class2/kernel/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE/source_separation_model/conv2d_transpose_2/bias+conv3Class2/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1

;0
<1
 
?
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
=	variables
>trainable_variables
?regularization_losses
tr
VARIABLE_VALUE1source_separation_model/conv2d_transpose_3/kernel-conv4Class2/kernel/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE/source_separation_model/conv2d_transpose_3/bias+conv4Class2/bias/.ATTRIBUTES/VARIABLE_VALUE

A0
B1

A0
B1
 
?
~non_trainable_variables

layers
?metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
a_
VARIABLE_VALUE&source_separation_model/dense_3/kernel%out/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE$source_separation_model/dense_3/bias#out/bias/.ATTRIBUTES/VARIABLE_VALUE

K0
L1

K0
L1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
 
N
0
1
2
3
4
5
6
7
	8

9
10
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_input_1Placeholder*0
_output_shapes
:?????????z?*
dtype0*%
shape:?????????z?
?	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1%source_separation_model/conv2d/kernel#source_separation_model/conv2d/bias'source_separation_model/conv2d_1/kernel%source_separation_model/conv2d_1/bias$source_separation_model/dense/kernel"source_separation_model/dense/bias&source_separation_model/dense_1/kernel$source_separation_model/dense_1/bias/source_separation_model/conv2d_transpose/kernel-source_separation_model/conv2d_transpose/bias1source_separation_model/conv2d_transpose_1/kernel/source_separation_model/conv2d_transpose_1/bias&source_separation_model/dense_2/kernel$source_separation_model/dense_2/bias1source_separation_model/conv2d_transpose_2/kernel/source_separation_model/conv2d_transpose_2/bias1source_separation_model/conv2d_transpose_3/kernel/source_separation_model/conv2d_transpose_3/bias&source_separation_model/dense_3/kernel$source_separation_model/dense_3/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_6468545
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename9source_separation_model/conv2d/kernel/Read/ReadVariableOp7source_separation_model/conv2d/bias/Read/ReadVariableOp;source_separation_model/conv2d_1/kernel/Read/ReadVariableOp9source_separation_model/conv2d_1/bias/Read/ReadVariableOp8source_separation_model/dense/kernel/Read/ReadVariableOp6source_separation_model/dense/bias/Read/ReadVariableOp:source_separation_model/dense_1/kernel/Read/ReadVariableOp8source_separation_model/dense_1/bias/Read/ReadVariableOp:source_separation_model/dense_2/kernel/Read/ReadVariableOp8source_separation_model/dense_2/bias/Read/ReadVariableOpCsource_separation_model/conv2d_transpose/kernel/Read/ReadVariableOpAsource_separation_model/conv2d_transpose/bias/Read/ReadVariableOpEsource_separation_model/conv2d_transpose_1/kernel/Read/ReadVariableOpCsource_separation_model/conv2d_transpose_1/bias/Read/ReadVariableOpEsource_separation_model/conv2d_transpose_2/kernel/Read/ReadVariableOpCsource_separation_model/conv2d_transpose_2/bias/Read/ReadVariableOpEsource_separation_model/conv2d_transpose_3/kernel/Read/ReadVariableOpCsource_separation_model/conv2d_transpose_3/bias/Read/ReadVariableOp:source_separation_model/dense_3/kernel/Read/ReadVariableOp8source_separation_model/dense_3/bias/Read/ReadVariableOpConst*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_6469398
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename%source_separation_model/conv2d/kernel#source_separation_model/conv2d/bias'source_separation_model/conv2d_1/kernel%source_separation_model/conv2d_1/bias$source_separation_model/dense/kernel"source_separation_model/dense/bias&source_separation_model/dense_1/kernel$source_separation_model/dense_1/bias&source_separation_model/dense_2/kernel$source_separation_model/dense_2/bias/source_separation_model/conv2d_transpose/kernel-source_separation_model/conv2d_transpose/bias1source_separation_model/conv2d_transpose_1/kernel/source_separation_model/conv2d_transpose_1/bias1source_separation_model/conv2d_transpose_2/kernel/source_separation_model/conv2d_transpose_2/bias1source_separation_model/conv2d_transpose_3/kernel/source_separation_model/conv2d_transpose_3/bias&source_separation_model/dense_3/kernel$source_separation_model/dense_3/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_6469468??
?;
?

T__inference_source_separation_model_layer_call_and_return_conditional_losses_6468252
x)
conv2d_6467961:?
conv2d_6467963:*
conv2d_1_6467977:
conv2d_1_6467979: 
dense_6468014:	?
dense_6468016:	?"
dense_1_6468051:	?
dense_1_6468053:2
conv2d_transpose_6468079:&
conv2d_transpose_6468081:5
conv2d_transpose_1_6468107:?(
conv2d_transpose_1_6468109:"
dense_2_6468144:	?
dense_2_6468146:4
conv2d_transpose_2_6468172:(
conv2d_transpose_2_6468174:5
conv2d_transpose_3_6468200:?(
conv2d_transpose_3_6468202:!
dense_3_6468246:<
dense_3_6468248:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?*conv2d_transpose_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallxconv2d_6467961conv2d_6467963*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_6467960?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_6467977conv2d_1_6467979*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6467976?
dense/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0dense_6468014dense_6468016*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????]?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_6468013?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_6468051dense_1_6468053*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_6468050?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0conv2d_transpose_6468079conv2d_transpose_6468081*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6468078?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_6468107conv2d_transpose_1_6468109*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_6468106?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_2_6468144dense_2_6468146*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_6468143?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0conv2d_transpose_2_6468172conv2d_transpose_2_6468174*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_6468171?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_6468200conv2d_transpose_3_6468202*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_6468199?
concatenate/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:03conv2d_transpose_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_6468212?
dense_3/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_3_6468246dense_3_6468248*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_6468245?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????z??
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????z?: : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:S O
0
_output_shapes
:?????????z?

_user_specified_namex
?
?
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_6468199

inputsC
(conv2d_transpose_readvariableop_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
stack/1Const*
_output_shapes
: *
dtype0*
value	B :zJ
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:?????????z?*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????z??
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????z
 
_user_specified_nameinputs
?
?
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_6468106

inputsC
(conv2d_transpose_readvariableop_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
stack/1Const*
_output_shapes
: *
dtype0*
value	B :zJ
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:?????????z?*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????z??
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????z
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_2_layer_call_fn_6469115

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_6467888?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_dense_3_layer_call_and_return_conditional_losses_6468245

inputs3
!tensordot_readvariableop_resource:<-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:<*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:~
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:?????????z?<?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????z?r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????z?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????z?z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????z?<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:X T
0
_output_shapes
:?????????z?<
 
_user_specified_nameinputs
?

?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6467976

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????]w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????z
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_layer_call_fn_6468959

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6467792?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?#
?
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_6467936

inputsC
(conv2d_transpose_readvariableop_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
add_1/yConst*
_output_shapes
: *
dtype0*
value
B :?L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_2_layer_call_fn_6469124

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_6468171w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????]: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
(__inference_conv2d_layer_call_fn_6468801

inputs"
unknown:?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_6467960w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????z?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????z?
 
_user_specified_nameinputs
?6
?
 __inference__traced_save_6469398
file_prefixD
@savev2_source_separation_model_conv2d_kernel_read_readvariableopB
>savev2_source_separation_model_conv2d_bias_read_readvariableopF
Bsavev2_source_separation_model_conv2d_1_kernel_read_readvariableopD
@savev2_source_separation_model_conv2d_1_bias_read_readvariableopC
?savev2_source_separation_model_dense_kernel_read_readvariableopA
=savev2_source_separation_model_dense_bias_read_readvariableopE
Asavev2_source_separation_model_dense_1_kernel_read_readvariableopC
?savev2_source_separation_model_dense_1_bias_read_readvariableopE
Asavev2_source_separation_model_dense_2_kernel_read_readvariableopC
?savev2_source_separation_model_dense_2_bias_read_readvariableopN
Jsavev2_source_separation_model_conv2d_transpose_kernel_read_readvariableopL
Hsavev2_source_separation_model_conv2d_transpose_bias_read_readvariableopP
Lsavev2_source_separation_model_conv2d_transpose_1_kernel_read_readvariableopN
Jsavev2_source_separation_model_conv2d_transpose_1_bias_read_readvariableopP
Lsavev2_source_separation_model_conv2d_transpose_2_kernel_read_readvariableopN
Jsavev2_source_separation_model_conv2d_transpose_2_bias_read_readvariableopP
Lsavev2_source_separation_model_conv2d_transpose_3_kernel_read_readvariableopN
Jsavev2_source_separation_model_conv2d_transpose_3_bias_read_readvariableopE
Asavev2_source_separation_model_dense_3_kernel_read_readvariableopC
?savev2_source_separation_model_dense_3_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB)dClass1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dClass1/bias/.ATTRIBUTES/VARIABLE_VALUEB)dClass2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dClass2/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv3Class1/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv3Class1/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv4Class1/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv4Class1/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv3Class2/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv3Class2/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv4Class2/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv4Class2/bias/.ATTRIBUTES/VARIABLE_VALUEB%out/kernel/.ATTRIBUTES/VARIABLE_VALUEB#out/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0@savev2_source_separation_model_conv2d_kernel_read_readvariableop>savev2_source_separation_model_conv2d_bias_read_readvariableopBsavev2_source_separation_model_conv2d_1_kernel_read_readvariableop@savev2_source_separation_model_conv2d_1_bias_read_readvariableop?savev2_source_separation_model_dense_kernel_read_readvariableop=savev2_source_separation_model_dense_bias_read_readvariableopAsavev2_source_separation_model_dense_1_kernel_read_readvariableop?savev2_source_separation_model_dense_1_bias_read_readvariableopAsavev2_source_separation_model_dense_2_kernel_read_readvariableop?savev2_source_separation_model_dense_2_bias_read_readvariableopJsavev2_source_separation_model_conv2d_transpose_kernel_read_readvariableopHsavev2_source_separation_model_conv2d_transpose_bias_read_readvariableopLsavev2_source_separation_model_conv2d_transpose_1_kernel_read_readvariableopJsavev2_source_separation_model_conv2d_transpose_1_bias_read_readvariableopLsavev2_source_separation_model_conv2d_transpose_2_kernel_read_readvariableopJsavev2_source_separation_model_conv2d_transpose_2_bias_read_readvariableopLsavev2_source_separation_model_conv2d_transpose_3_kernel_read_readvariableopJsavev2_source_separation_model_conv2d_transpose_3_bias_read_readvariableopAsavev2_source_separation_model_dense_3_kernel_read_readvariableop?savev2_source_separation_model_dense_3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *#
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :?::::	?:?:	?::	?::::?::::?::<:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:?: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%	!

_output_shapes
:	?: 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::-)
'
_output_shapes
:?: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::-)
'
_output_shapes
:?: 

_output_shapes
::$ 

_output_shapes

:<: 

_output_shapes
::

_output_shapes
: 
?
?
D__inference_dense_1_layer_call_and_return_conditional_losses_6468050

inputs4
!tensordot_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:~
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:?????????]??
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????]r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????]i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????]z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????]?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:X T
0
_output_shapes
:?????????]?
 
_user_specified_nameinputs
?
?
D__inference_dense_2_layer_call_and_return_conditional_losses_6468143

inputs4
!tensordot_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:~
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:?????????]??
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????]r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????]i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????]z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????]?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:X T
0
_output_shapes
:?????????]?
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_6468545
input_1"
unknown:?
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	?
	unknown_4:	?
	unknown_5:	?
	unknown_6:#
	unknown_7:
	unknown_8:$
	unknown_9:?

unknown_10:

unknown_11:	?

unknown_12:$

unknown_13:

unknown_14:%

unknown_15:?

unknown_16:

unknown_17:<

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_6467751x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????z?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????z?: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:?????????z?
!
_user_specified_name	input_1
?
?
D__inference_dense_3_layer_call_and_return_conditional_losses_6469315

inputs3
!tensordot_readvariableop_resource:<-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:<*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:~
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:?????????z?<?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????z?r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????z?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????z?z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????z?<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:X T
0
_output_shapes
:?????????z?<
 
_user_specified_nameinputs
?#
?
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_6469083

inputsC
(conv2d_transpose_readvariableop_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
add_1/yConst*
_output_shapes
: *
dtype0*
value
B :?L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_dense_layer_call_fn_6468839

inputs
unknown:	?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????]?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_6468013x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????]?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????]: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????]
 
_user_specified_nameinputs
?

?
C__inference_conv2d_layer_call_and_return_conditional_losses_6467960

inputs9
conv2d_readvariableop_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????zg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????zw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????z?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????z?
 
_user_specified_nameinputs
?
?
B__inference_dense_layer_call_and_return_conditional_losses_6468870

inputs4
!tensordot_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:}
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:?????????]?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????]?s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????]?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????]?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????]?z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:W S
/
_output_shapes
:?????????]
 
_user_specified_nameinputs
?#
?
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6469005

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_dense_2_layer_call_fn_6468919

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_6468143w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????]`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????]?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????]?
 
_user_specified_nameinputs
?
r
H__inference_concatenate_layer_call_and_return_conditional_losses_6468212

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :~
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????z?<`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????z?<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????z?:?????????z?:X T
0
_output_shapes
:?????????z?
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????z?
 
_user_specified_nameinputs
?#
?
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_6469239

inputsC
(conv2d_transpose_readvariableop_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
add_1/yConst*
_output_shapes
: *
dtype0*
value
B :?L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_1_layer_call_fn_6469046

inputs"
unknown:?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_6468106x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????z?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????z: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????z
 
_user_specified_nameinputs
?;
?

T__inference_source_separation_model_layer_call_and_return_conditional_losses_6468498
input_1)
conv2d_6468446:?
conv2d_6468448:*
conv2d_1_6468451:
conv2d_1_6468453: 
dense_6468456:	?
dense_6468458:	?"
dense_1_6468461:	?
dense_1_6468463:2
conv2d_transpose_6468466:&
conv2d_transpose_6468468:5
conv2d_transpose_1_6468471:?(
conv2d_transpose_1_6468473:"
dense_2_6468476:	?
dense_2_6468478:4
conv2d_transpose_2_6468481:(
conv2d_transpose_2_6468483:5
conv2d_transpose_3_6468486:?(
conv2d_transpose_3_6468488:!
dense_3_6468492:<
dense_3_6468494:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?*conv2d_transpose_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_6468446conv2d_6468448*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_6467960?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_6468451conv2d_1_6468453*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6467976?
dense/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0dense_6468456dense_6468458*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????]?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_6468013?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_6468461dense_1_6468463*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_6468050?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0conv2d_transpose_6468466conv2d_transpose_6468468*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6468078?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_6468471conv2d_transpose_1_6468473*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_6468106?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_2_6468476dense_2_6468478*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_6468143?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0conv2d_transpose_2_6468481conv2d_transpose_2_6468483*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_6468171?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_6468486conv2d_transpose_3_6468488*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_6468199?
concatenate/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:03conv2d_transpose_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_6468212?
dense_3/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_3_6468492dense_3_6468494*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_6468245?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????z??
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????z?: : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:Y U
0
_output_shapes
:?????????z?
!
_user_specified_name	input_1
?
?
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_6469106

inputsC
(conv2d_transpose_readvariableop_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
stack/1Const*
_output_shapes
: *
dtype0*
value	B :zJ
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:?????????z?*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????z??
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????z
 
_user_specified_nameinputs
?

?
C__inference_conv2d_layer_call_and_return_conditional_losses_6468811

inputs9
conv2d_readvariableop_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????zg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????zw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????z?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????z?
 
_user_specified_nameinputs
?

?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6468830

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????]w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????z
 
_user_specified_nameinputs
?
?
)__inference_dense_3_layer_call_fn_6469284

inputs
unknown:<
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_6468245x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????z?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????z?<: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????z?<
 
_user_specified_nameinputs
?
Y
-__inference_concatenate_layer_call_fn_6469268
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_6468212i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????z?<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????z?:?????????z?:Z V
0
_output_shapes
:?????????z?
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????z?
"
_user_specified_name
inputs/1
?
?
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_6468171

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
stack/1Const*
_output_shapes
: *
dtype0*
value	B :zI
stack/2Const*
_output_shapes
: *
dtype0*
value	B :I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????zg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????z?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_layer_call_fn_6468968

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6468078w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????]: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????]
 
_user_specified_nameinputs
?#
?
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_6467840

inputsC
(conv2d_transpose_readvariableop_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
add_1/yConst*
_output_shapes
: *
dtype0*
value
B :?L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
t
H__inference_concatenate_layer_call_and_return_conditional_losses_6469275
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????z?<`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????z?<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????z?:?????????z?:Z V
0
_output_shapes
:?????????z?
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????z?
"
_user_specified_name
inputs/1
??
?
"__inference__wrapped_model_6467751
input_1X
=source_separation_model_conv2d_conv2d_readvariableop_resource:?L
>source_separation_model_conv2d_biasadd_readvariableop_resource:Y
?source_separation_model_conv2d_1_conv2d_readvariableop_resource:N
@source_separation_model_conv2d_1_biasadd_readvariableop_resource:R
?source_separation_model_dense_tensordot_readvariableop_resource:	?L
=source_separation_model_dense_biasadd_readvariableop_resource:	?T
Asource_separation_model_dense_1_tensordot_readvariableop_resource:	?M
?source_separation_model_dense_1_biasadd_readvariableop_resource:k
Qsource_separation_model_conv2d_transpose_conv2d_transpose_readvariableop_resource:V
Hsource_separation_model_conv2d_transpose_biasadd_readvariableop_resource:n
Ssource_separation_model_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:?X
Jsource_separation_model_conv2d_transpose_1_biasadd_readvariableop_resource:T
Asource_separation_model_dense_2_tensordot_readvariableop_resource:	?M
?source_separation_model_dense_2_biasadd_readvariableop_resource:m
Ssource_separation_model_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:X
Jsource_separation_model_conv2d_transpose_2_biasadd_readvariableop_resource:n
Ssource_separation_model_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:?X
Jsource_separation_model_conv2d_transpose_3_biasadd_readvariableop_resource:S
Asource_separation_model_dense_3_tensordot_readvariableop_resource:<M
?source_separation_model_dense_3_biasadd_readvariableop_resource:
identity??5source_separation_model/conv2d/BiasAdd/ReadVariableOp?4source_separation_model/conv2d/Conv2D/ReadVariableOp?7source_separation_model/conv2d_1/BiasAdd/ReadVariableOp?6source_separation_model/conv2d_1/Conv2D/ReadVariableOp??source_separation_model/conv2d_transpose/BiasAdd/ReadVariableOp?Hsource_separation_model/conv2d_transpose/conv2d_transpose/ReadVariableOp?Asource_separation_model/conv2d_transpose_1/BiasAdd/ReadVariableOp?Jsource_separation_model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?Asource_separation_model/conv2d_transpose_2/BiasAdd/ReadVariableOp?Jsource_separation_model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?Asource_separation_model/conv2d_transpose_3/BiasAdd/ReadVariableOp?Jsource_separation_model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?4source_separation_model/dense/BiasAdd/ReadVariableOp?6source_separation_model/dense/Tensordot/ReadVariableOp?6source_separation_model/dense_1/BiasAdd/ReadVariableOp?8source_separation_model/dense_1/Tensordot/ReadVariableOp?6source_separation_model/dense_2/BiasAdd/ReadVariableOp?8source_separation_model/dense_2/Tensordot/ReadVariableOp?6source_separation_model/dense_3/BiasAdd/ReadVariableOp?8source_separation_model/dense_3/Tensordot/ReadVariableOp?
4source_separation_model/conv2d/Conv2D/ReadVariableOpReadVariableOp=source_separation_model_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
%source_separation_model/conv2d/Conv2DConv2Dinput_1<source_separation_model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
?
5source_separation_model/conv2d/BiasAdd/ReadVariableOpReadVariableOp>source_separation_model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
&source_separation_model/conv2d/BiasAddBiasAdd.source_separation_model/conv2d/Conv2D:output:0=source_separation_model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z?
6source_separation_model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp?source_separation_model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
'source_separation_model/conv2d_1/Conv2DConv2D/source_separation_model/conv2d/BiasAdd:output:0>source_separation_model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]*
paddingVALID*
strides
?
7source_separation_model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp@source_separation_model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
(source_separation_model/conv2d_1/BiasAddBiasAdd0source_separation_model/conv2d_1/Conv2D:output:0?source_separation_model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]?
6source_separation_model/dense/Tensordot/ReadVariableOpReadVariableOp?source_separation_model_dense_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0v
,source_separation_model/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
,source_separation_model/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          ?
-source_separation_model/dense/Tensordot/ShapeShape1source_separation_model/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:w
5source_separation_model/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0source_separation_model/dense/Tensordot/GatherV2GatherV26source_separation_model/dense/Tensordot/Shape:output:05source_separation_model/dense/Tensordot/free:output:0>source_separation_model/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7source_separation_model/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2source_separation_model/dense/Tensordot/GatherV2_1GatherV26source_separation_model/dense/Tensordot/Shape:output:05source_separation_model/dense/Tensordot/axes:output:0@source_separation_model/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-source_separation_model/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
,source_separation_model/dense/Tensordot/ProdProd9source_separation_model/dense/Tensordot/GatherV2:output:06source_separation_model/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/source_separation_model/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
.source_separation_model/dense/Tensordot/Prod_1Prod;source_separation_model/dense/Tensordot/GatherV2_1:output:08source_separation_model/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3source_separation_model/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
.source_separation_model/dense/Tensordot/concatConcatV25source_separation_model/dense/Tensordot/free:output:05source_separation_model/dense/Tensordot/axes:output:0<source_separation_model/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
-source_separation_model/dense/Tensordot/stackPack5source_separation_model/dense/Tensordot/Prod:output:07source_separation_model/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
1source_separation_model/dense/Tensordot/transpose	Transpose1source_separation_model/conv2d_1/BiasAdd:output:07source_separation_model/dense/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????]?
/source_separation_model/dense/Tensordot/ReshapeReshape5source_separation_model/dense/Tensordot/transpose:y:06source_separation_model/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
.source_separation_model/dense/Tensordot/MatMulMatMul8source_separation_model/dense/Tensordot/Reshape:output:0>source_separation_model/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????z
/source_separation_model/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?w
5source_separation_model/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0source_separation_model/dense/Tensordot/concat_1ConcatV29source_separation_model/dense/Tensordot/GatherV2:output:08source_separation_model/dense/Tensordot/Const_2:output:0>source_separation_model/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
'source_separation_model/dense/TensordotReshape8source_separation_model/dense/Tensordot/MatMul:product:09source_separation_model/dense/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????]??
4source_separation_model/dense/BiasAdd/ReadVariableOpReadVariableOp=source_separation_model_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
%source_separation_model/dense/BiasAddBiasAdd0source_separation_model/dense/Tensordot:output:0<source_separation_model/dense/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????]??
"source_separation_model/dense/ReluRelu.source_separation_model/dense/BiasAdd:output:0*
T0*0
_output_shapes
:?????????]??
8source_separation_model/dense_1/Tensordot/ReadVariableOpReadVariableOpAsource_separation_model_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0x
.source_separation_model/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
.source_separation_model/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          ?
/source_separation_model/dense_1/Tensordot/ShapeShape0source_separation_model/dense/Relu:activations:0*
T0*
_output_shapes
:y
7source_separation_model/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2source_separation_model/dense_1/Tensordot/GatherV2GatherV28source_separation_model/dense_1/Tensordot/Shape:output:07source_separation_model/dense_1/Tensordot/free:output:0@source_separation_model/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9source_separation_model/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4source_separation_model/dense_1/Tensordot/GatherV2_1GatherV28source_separation_model/dense_1/Tensordot/Shape:output:07source_separation_model/dense_1/Tensordot/axes:output:0Bsource_separation_model/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/source_separation_model/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.source_separation_model/dense_1/Tensordot/ProdProd;source_separation_model/dense_1/Tensordot/GatherV2:output:08source_separation_model/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1source_separation_model/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0source_separation_model/dense_1/Tensordot/Prod_1Prod=source_separation_model/dense_1/Tensordot/GatherV2_1:output:0:source_separation_model/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5source_separation_model/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0source_separation_model/dense_1/Tensordot/concatConcatV27source_separation_model/dense_1/Tensordot/free:output:07source_separation_model/dense_1/Tensordot/axes:output:0>source_separation_model/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/source_separation_model/dense_1/Tensordot/stackPack7source_separation_model/dense_1/Tensordot/Prod:output:09source_separation_model/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3source_separation_model/dense_1/Tensordot/transpose	Transpose0source_separation_model/dense/Relu:activations:09source_separation_model/dense_1/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????]??
1source_separation_model/dense_1/Tensordot/ReshapeReshape7source_separation_model/dense_1/Tensordot/transpose:y:08source_separation_model/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0source_separation_model/dense_1/Tensordot/MatMulMatMul:source_separation_model/dense_1/Tensordot/Reshape:output:0@source_separation_model/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1source_separation_model/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7source_separation_model/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2source_separation_model/dense_1/Tensordot/concat_1ConcatV2;source_separation_model/dense_1/Tensordot/GatherV2:output:0:source_separation_model/dense_1/Tensordot/Const_2:output:0@source_separation_model/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)source_separation_model/dense_1/TensordotReshape:source_separation_model/dense_1/Tensordot/MatMul:product:0;source_separation_model/dense_1/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????]?
6source_separation_model/dense_1/BiasAdd/ReadVariableOpReadVariableOp?source_separation_model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'source_separation_model/dense_1/BiasAddBiasAdd2source_separation_model/dense_1/Tensordot:output:0>source_separation_model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]?
$source_separation_model/dense_1/ReluRelu0source_separation_model/dense_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????]?
.source_separation_model/conv2d_transpose/ShapeShape2source_separation_model/dense_1/Relu:activations:0*
T0*
_output_shapes
:?
<source_separation_model/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
>source_separation_model/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
>source_separation_model/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6source_separation_model/conv2d_transpose/strided_sliceStridedSlice7source_separation_model/conv2d_transpose/Shape:output:0Esource_separation_model/conv2d_transpose/strided_slice/stack:output:0Gsource_separation_model/conv2d_transpose/strided_slice/stack_1:output:0Gsource_separation_model/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0source_separation_model/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :zr
0source_separation_model/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :r
0source_separation_model/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
.source_separation_model/conv2d_transpose/stackPack?source_separation_model/conv2d_transpose/strided_slice:output:09source_separation_model/conv2d_transpose/stack/1:output:09source_separation_model/conv2d_transpose/stack/2:output:09source_separation_model/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:?
>source_separation_model/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@source_separation_model/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@source_separation_model/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8source_separation_model/conv2d_transpose/strided_slice_1StridedSlice7source_separation_model/conv2d_transpose/stack:output:0Gsource_separation_model/conv2d_transpose/strided_slice_1/stack:output:0Isource_separation_model/conv2d_transpose/strided_slice_1/stack_1:output:0Isource_separation_model/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Hsource_separation_model/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpQsource_separation_model_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
9source_separation_model/conv2d_transpose/conv2d_transposeConv2DBackpropInput7source_separation_model/conv2d_transpose/stack:output:0Psource_separation_model/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:02source_separation_model/dense_1/Relu:activations:0*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
?
?source_separation_model/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpHsource_separation_model_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
0source_separation_model/conv2d_transpose/BiasAddBiasAddBsource_separation_model/conv2d_transpose/conv2d_transpose:output:0Gsource_separation_model/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z?
0source_separation_model/conv2d_transpose_1/ShapeShape9source_separation_model/conv2d_transpose/BiasAdd:output:0*
T0*
_output_shapes
:?
>source_separation_model/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@source_separation_model/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@source_separation_model/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8source_separation_model/conv2d_transpose_1/strided_sliceStridedSlice9source_separation_model/conv2d_transpose_1/Shape:output:0Gsource_separation_model/conv2d_transpose_1/strided_slice/stack:output:0Isource_separation_model/conv2d_transpose_1/strided_slice/stack_1:output:0Isource_separation_model/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2source_separation_model/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :zu
2source_separation_model/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?t
2source_separation_model/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
0source_separation_model/conv2d_transpose_1/stackPackAsource_separation_model/conv2d_transpose_1/strided_slice:output:0;source_separation_model/conv2d_transpose_1/stack/1:output:0;source_separation_model/conv2d_transpose_1/stack/2:output:0;source_separation_model/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:?
@source_separation_model/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bsource_separation_model/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bsource_separation_model/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:source_separation_model/conv2d_transpose_1/strided_slice_1StridedSlice9source_separation_model/conv2d_transpose_1/stack:output:0Isource_separation_model/conv2d_transpose_1/strided_slice_1/stack:output:0Ksource_separation_model/conv2d_transpose_1/strided_slice_1/stack_1:output:0Ksource_separation_model/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Jsource_separation_model/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpSsource_separation_model_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
;source_separation_model/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput9source_separation_model/conv2d_transpose_1/stack:output:0Rsource_separation_model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:09source_separation_model/conv2d_transpose/BiasAdd:output:0*
T0*0
_output_shapes
:?????????z?*
paddingVALID*
strides
?
Asource_separation_model/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpJsource_separation_model_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
2source_separation_model/conv2d_transpose_1/BiasAddBiasAddDsource_separation_model/conv2d_transpose_1/conv2d_transpose:output:0Isource_separation_model/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z??
8source_separation_model/dense_2/Tensordot/ReadVariableOpReadVariableOpAsource_separation_model_dense_2_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0x
.source_separation_model/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
.source_separation_model/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          ?
/source_separation_model/dense_2/Tensordot/ShapeShape0source_separation_model/dense/Relu:activations:0*
T0*
_output_shapes
:y
7source_separation_model/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2source_separation_model/dense_2/Tensordot/GatherV2GatherV28source_separation_model/dense_2/Tensordot/Shape:output:07source_separation_model/dense_2/Tensordot/free:output:0@source_separation_model/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9source_separation_model/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4source_separation_model/dense_2/Tensordot/GatherV2_1GatherV28source_separation_model/dense_2/Tensordot/Shape:output:07source_separation_model/dense_2/Tensordot/axes:output:0Bsource_separation_model/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/source_separation_model/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.source_separation_model/dense_2/Tensordot/ProdProd;source_separation_model/dense_2/Tensordot/GatherV2:output:08source_separation_model/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1source_separation_model/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0source_separation_model/dense_2/Tensordot/Prod_1Prod=source_separation_model/dense_2/Tensordot/GatherV2_1:output:0:source_separation_model/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5source_separation_model/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0source_separation_model/dense_2/Tensordot/concatConcatV27source_separation_model/dense_2/Tensordot/free:output:07source_separation_model/dense_2/Tensordot/axes:output:0>source_separation_model/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/source_separation_model/dense_2/Tensordot/stackPack7source_separation_model/dense_2/Tensordot/Prod:output:09source_separation_model/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3source_separation_model/dense_2/Tensordot/transpose	Transpose0source_separation_model/dense/Relu:activations:09source_separation_model/dense_2/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????]??
1source_separation_model/dense_2/Tensordot/ReshapeReshape7source_separation_model/dense_2/Tensordot/transpose:y:08source_separation_model/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0source_separation_model/dense_2/Tensordot/MatMulMatMul:source_separation_model/dense_2/Tensordot/Reshape:output:0@source_separation_model/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1source_separation_model/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7source_separation_model/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2source_separation_model/dense_2/Tensordot/concat_1ConcatV2;source_separation_model/dense_2/Tensordot/GatherV2:output:0:source_separation_model/dense_2/Tensordot/Const_2:output:0@source_separation_model/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)source_separation_model/dense_2/TensordotReshape:source_separation_model/dense_2/Tensordot/MatMul:product:0;source_separation_model/dense_2/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????]?
6source_separation_model/dense_2/BiasAdd/ReadVariableOpReadVariableOp?source_separation_model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'source_separation_model/dense_2/BiasAddBiasAdd2source_separation_model/dense_2/Tensordot:output:0>source_separation_model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]?
$source_separation_model/dense_2/ReluRelu0source_separation_model/dense_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????]?
0source_separation_model/conv2d_transpose_2/ShapeShape2source_separation_model/dense_2/Relu:activations:0*
T0*
_output_shapes
:?
>source_separation_model/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@source_separation_model/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@source_separation_model/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8source_separation_model/conv2d_transpose_2/strided_sliceStridedSlice9source_separation_model/conv2d_transpose_2/Shape:output:0Gsource_separation_model/conv2d_transpose_2/strided_slice/stack:output:0Isource_separation_model/conv2d_transpose_2/strided_slice/stack_1:output:0Isource_separation_model/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2source_separation_model/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :zt
2source_separation_model/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :t
2source_separation_model/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
0source_separation_model/conv2d_transpose_2/stackPackAsource_separation_model/conv2d_transpose_2/strided_slice:output:0;source_separation_model/conv2d_transpose_2/stack/1:output:0;source_separation_model/conv2d_transpose_2/stack/2:output:0;source_separation_model/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:?
@source_separation_model/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bsource_separation_model/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bsource_separation_model/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:source_separation_model/conv2d_transpose_2/strided_slice_1StridedSlice9source_separation_model/conv2d_transpose_2/stack:output:0Isource_separation_model/conv2d_transpose_2/strided_slice_1/stack:output:0Ksource_separation_model/conv2d_transpose_2/strided_slice_1/stack_1:output:0Ksource_separation_model/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Jsource_separation_model/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpSsource_separation_model_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
;source_separation_model/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput9source_separation_model/conv2d_transpose_2/stack:output:0Rsource_separation_model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:02source_separation_model/dense_2/Relu:activations:0*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
?
Asource_separation_model/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpJsource_separation_model_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
2source_separation_model/conv2d_transpose_2/BiasAddBiasAddDsource_separation_model/conv2d_transpose_2/conv2d_transpose:output:0Isource_separation_model/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z?
0source_separation_model/conv2d_transpose_3/ShapeShape;source_separation_model/conv2d_transpose_2/BiasAdd:output:0*
T0*
_output_shapes
:?
>source_separation_model/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@source_separation_model/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@source_separation_model/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8source_separation_model/conv2d_transpose_3/strided_sliceStridedSlice9source_separation_model/conv2d_transpose_3/Shape:output:0Gsource_separation_model/conv2d_transpose_3/strided_slice/stack:output:0Isource_separation_model/conv2d_transpose_3/strided_slice/stack_1:output:0Isource_separation_model/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2source_separation_model/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :zu
2source_separation_model/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?t
2source_separation_model/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
0source_separation_model/conv2d_transpose_3/stackPackAsource_separation_model/conv2d_transpose_3/strided_slice:output:0;source_separation_model/conv2d_transpose_3/stack/1:output:0;source_separation_model/conv2d_transpose_3/stack/2:output:0;source_separation_model/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:?
@source_separation_model/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bsource_separation_model/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bsource_separation_model/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:source_separation_model/conv2d_transpose_3/strided_slice_1StridedSlice9source_separation_model/conv2d_transpose_3/stack:output:0Isource_separation_model/conv2d_transpose_3/strided_slice_1/stack:output:0Ksource_separation_model/conv2d_transpose_3/strided_slice_1/stack_1:output:0Ksource_separation_model/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Jsource_separation_model/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpSsource_separation_model_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
;source_separation_model/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput9source_separation_model/conv2d_transpose_3/stack:output:0Rsource_separation_model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0;source_separation_model/conv2d_transpose_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????z?*
paddingVALID*
strides
?
Asource_separation_model/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpJsource_separation_model_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
2source_separation_model/conv2d_transpose_3/BiasAddBiasAddDsource_separation_model/conv2d_transpose_3/conv2d_transpose:output:0Isource_separation_model/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z?q
/source_separation_model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
*source_separation_model/concatenate/concatConcatV2;source_separation_model/conv2d_transpose_1/BiasAdd:output:0;source_separation_model/conv2d_transpose_3/BiasAdd:output:08source_separation_model/concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????z?<?
8source_separation_model/dense_3/Tensordot/ReadVariableOpReadVariableOpAsource_separation_model_dense_3_tensordot_readvariableop_resource*
_output_shapes

:<*
dtype0x
.source_separation_model/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
.source_separation_model/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          ?
/source_separation_model/dense_3/Tensordot/ShapeShape3source_separation_model/concatenate/concat:output:0*
T0*
_output_shapes
:y
7source_separation_model/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2source_separation_model/dense_3/Tensordot/GatherV2GatherV28source_separation_model/dense_3/Tensordot/Shape:output:07source_separation_model/dense_3/Tensordot/free:output:0@source_separation_model/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9source_separation_model/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4source_separation_model/dense_3/Tensordot/GatherV2_1GatherV28source_separation_model/dense_3/Tensordot/Shape:output:07source_separation_model/dense_3/Tensordot/axes:output:0Bsource_separation_model/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/source_separation_model/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.source_separation_model/dense_3/Tensordot/ProdProd;source_separation_model/dense_3/Tensordot/GatherV2:output:08source_separation_model/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1source_separation_model/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0source_separation_model/dense_3/Tensordot/Prod_1Prod=source_separation_model/dense_3/Tensordot/GatherV2_1:output:0:source_separation_model/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5source_separation_model/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0source_separation_model/dense_3/Tensordot/concatConcatV27source_separation_model/dense_3/Tensordot/free:output:07source_separation_model/dense_3/Tensordot/axes:output:0>source_separation_model/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/source_separation_model/dense_3/Tensordot/stackPack7source_separation_model/dense_3/Tensordot/Prod:output:09source_separation_model/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3source_separation_model/dense_3/Tensordot/transpose	Transpose3source_separation_model/concatenate/concat:output:09source_separation_model/dense_3/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????z?<?
1source_separation_model/dense_3/Tensordot/ReshapeReshape7source_separation_model/dense_3/Tensordot/transpose:y:08source_separation_model/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0source_separation_model/dense_3/Tensordot/MatMulMatMul:source_separation_model/dense_3/Tensordot/Reshape:output:0@source_separation_model/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1source_separation_model/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7source_separation_model/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2source_separation_model/dense_3/Tensordot/concat_1ConcatV2;source_separation_model/dense_3/Tensordot/GatherV2:output:0:source_separation_model/dense_3/Tensordot/Const_2:output:0@source_separation_model/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)source_separation_model/dense_3/TensordotReshape:source_separation_model/dense_3/Tensordot/MatMul:product:0;source_separation_model/dense_3/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????z??
6source_separation_model/dense_3/BiasAdd/ReadVariableOpReadVariableOp?source_separation_model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'source_separation_model/dense_3/BiasAddBiasAdd2source_separation_model/dense_3/Tensordot:output:0>source_separation_model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z??
$source_separation_model/dense_3/ReluRelu0source_separation_model/dense_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????z??
IdentityIdentity2source_separation_model/dense_3/Relu:activations:0^NoOp*
T0*0
_output_shapes
:?????????z??

NoOpNoOp6^source_separation_model/conv2d/BiasAdd/ReadVariableOp5^source_separation_model/conv2d/Conv2D/ReadVariableOp8^source_separation_model/conv2d_1/BiasAdd/ReadVariableOp7^source_separation_model/conv2d_1/Conv2D/ReadVariableOp@^source_separation_model/conv2d_transpose/BiasAdd/ReadVariableOpI^source_separation_model/conv2d_transpose/conv2d_transpose/ReadVariableOpB^source_separation_model/conv2d_transpose_1/BiasAdd/ReadVariableOpK^source_separation_model/conv2d_transpose_1/conv2d_transpose/ReadVariableOpB^source_separation_model/conv2d_transpose_2/BiasAdd/ReadVariableOpK^source_separation_model/conv2d_transpose_2/conv2d_transpose/ReadVariableOpB^source_separation_model/conv2d_transpose_3/BiasAdd/ReadVariableOpK^source_separation_model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp5^source_separation_model/dense/BiasAdd/ReadVariableOp7^source_separation_model/dense/Tensordot/ReadVariableOp7^source_separation_model/dense_1/BiasAdd/ReadVariableOp9^source_separation_model/dense_1/Tensordot/ReadVariableOp7^source_separation_model/dense_2/BiasAdd/ReadVariableOp9^source_separation_model/dense_2/Tensordot/ReadVariableOp7^source_separation_model/dense_3/BiasAdd/ReadVariableOp9^source_separation_model/dense_3/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????z?: : : : : : : : : : : : : : : : : : : : 2n
5source_separation_model/conv2d/BiasAdd/ReadVariableOp5source_separation_model/conv2d/BiasAdd/ReadVariableOp2l
4source_separation_model/conv2d/Conv2D/ReadVariableOp4source_separation_model/conv2d/Conv2D/ReadVariableOp2r
7source_separation_model/conv2d_1/BiasAdd/ReadVariableOp7source_separation_model/conv2d_1/BiasAdd/ReadVariableOp2p
6source_separation_model/conv2d_1/Conv2D/ReadVariableOp6source_separation_model/conv2d_1/Conv2D/ReadVariableOp2?
?source_separation_model/conv2d_transpose/BiasAdd/ReadVariableOp?source_separation_model/conv2d_transpose/BiasAdd/ReadVariableOp2?
Hsource_separation_model/conv2d_transpose/conv2d_transpose/ReadVariableOpHsource_separation_model/conv2d_transpose/conv2d_transpose/ReadVariableOp2?
Asource_separation_model/conv2d_transpose_1/BiasAdd/ReadVariableOpAsource_separation_model/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
Jsource_separation_model/conv2d_transpose_1/conv2d_transpose/ReadVariableOpJsource_separation_model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2?
Asource_separation_model/conv2d_transpose_2/BiasAdd/ReadVariableOpAsource_separation_model/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
Jsource_separation_model/conv2d_transpose_2/conv2d_transpose/ReadVariableOpJsource_separation_model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2?
Asource_separation_model/conv2d_transpose_3/BiasAdd/ReadVariableOpAsource_separation_model/conv2d_transpose_3/BiasAdd/ReadVariableOp2?
Jsource_separation_model/conv2d_transpose_3/conv2d_transpose/ReadVariableOpJsource_separation_model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2l
4source_separation_model/dense/BiasAdd/ReadVariableOp4source_separation_model/dense/BiasAdd/ReadVariableOp2p
6source_separation_model/dense/Tensordot/ReadVariableOp6source_separation_model/dense/Tensordot/ReadVariableOp2p
6source_separation_model/dense_1/BiasAdd/ReadVariableOp6source_separation_model/dense_1/BiasAdd/ReadVariableOp2t
8source_separation_model/dense_1/Tensordot/ReadVariableOp8source_separation_model/dense_1/Tensordot/ReadVariableOp2p
6source_separation_model/dense_2/BiasAdd/ReadVariableOp6source_separation_model/dense_2/BiasAdd/ReadVariableOp2t
8source_separation_model/dense_2/Tensordot/ReadVariableOp8source_separation_model/dense_2/Tensordot/ReadVariableOp2p
6source_separation_model/dense_3/BiasAdd/ReadVariableOp6source_separation_model/dense_3/BiasAdd/ReadVariableOp2t
8source_separation_model/dense_3/Tensordot/ReadVariableOp8source_separation_model/dense_3/Tensordot/ReadVariableOp:Y U
0
_output_shapes
:?????????z?
!
_user_specified_name	input_1
?
?
4__inference_conv2d_transpose_1_layer_call_fn_6469037

inputs"
unknown:?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_6467840?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_dense_2_layer_call_and_return_conditional_losses_6468950

inputs4
!tensordot_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:~
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:?????????]??
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????]r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????]i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????]z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????]?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:X T
0
_output_shapes
:?????????]?
 
_user_specified_nameinputs
?
?
9__inference_source_separation_model_layer_call_fn_6468295
input_1"
unknown:?
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	?
	unknown_4:	?
	unknown_5:	?
	unknown_6:#
	unknown_7:
	unknown_8:$
	unknown_9:?

unknown_10:

unknown_11:	?

unknown_12:$

unknown_13:

unknown_14:%

unknown_15:?

unknown_16:

unknown_17:<

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_source_separation_model_layer_call_and_return_conditional_losses_6468252x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????z?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????z?: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:?????????z?
!
_user_specified_name	input_1
?
?
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6469028

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
stack/1Const*
_output_shapes
: *
dtype0*
value	B :zI
stack/2Const*
_output_shapes
: *
dtype0*
value	B :I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????zg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????z?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_3_layer_call_fn_6469202

inputs"
unknown:?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_6468199x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????z?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????z: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????z
 
_user_specified_nameinputs
?
?
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6468078

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
stack/1Const*
_output_shapes
: *
dtype0*
value	B :zI
stack/2Const*
_output_shapes
: *
dtype0*
value	B :I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????zg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????z?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????]
 
_user_specified_nameinputs
?#
?
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6467792

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_dense_layer_call_and_return_conditional_losses_6468013

inputs4
!tensordot_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:}
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:?????????]?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????]?s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????]?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????]?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????]?z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:W S
/
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
*__inference_conv2d_1_layer_call_fn_6468820

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6467976w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????]`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????z: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????z
 
_user_specified_nameinputs
?
?
)__inference_dense_1_layer_call_fn_6468879

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_6468050w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????]`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????]?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????]?
 
_user_specified_nameinputs
?
?
D__inference_dense_1_layer_call_and_return_conditional_losses_6468910

inputs4
!tensordot_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:~
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:?????????]??
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????]r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????]i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????]z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????]?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:X T
0
_output_shapes
:?????????]?
 
_user_specified_nameinputs
?#
?
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_6469161

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?
T__inference_source_separation_model_layer_call_and_return_conditional_losses_6468792
x@
%conv2d_conv2d_readvariableop_resource:?4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource::
'dense_tensordot_readvariableop_resource:	?4
%dense_biasadd_readvariableop_resource:	?<
)dense_1_tensordot_readvariableop_resource:	?5
'dense_1_biasadd_readvariableop_resource:S
9conv2d_transpose_conv2d_transpose_readvariableop_resource:>
0conv2d_transpose_biasadd_readvariableop_resource:V
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:?@
2conv2d_transpose_1_biasadd_readvariableop_resource:<
)dense_2_tensordot_readvariableop_resource:	?5
'dense_2_biasadd_readvariableop_resource:U
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_2_biasadd_readvariableop_resource:V
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:?@
2conv2d_transpose_3_biasadd_readvariableop_resource:;
)dense_3_tensordot_readvariableop_resource:<5
'dense_3_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?)conv2d_transpose_2/BiasAdd/ReadVariableOp?2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?)conv2d_transpose_3/BiasAdd/ReadVariableOp?2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/Tensordot/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp? dense_3/Tensordot/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dconv2d/BiasAdd:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          ^
dense/Tensordot/ShapeShapeconv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/transpose	Transposeconv2d_1/BiasAdd:output:0dense/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????]?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????]?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????]?e

dense/ReluReludense/BiasAdd:output:0*
T0*0
_output_shapes
:?????????]??
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          _
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????]??
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????]?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]h
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????]`
conv2d_transpose/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :zZ
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0dense_1/Relu:activations:0*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????zi
conv2d_transpose_1/ShapeShape!conv2d_transpose/BiasAdd:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :z]
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0!conv2d_transpose/BiasAdd:output:0*
T0*0
_output_shapes
:?????????z?*
paddingVALID*
strides
?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z??
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0`
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          _
dense_2/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_2/Tensordot/transpose	Transposedense/Relu:activations:0!dense_2/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????]??
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????]?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]h
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????]b
conv2d_transpose_2/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :z\
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0dense_2/Relu:activations:0*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
?
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????zk
conv2d_transpose_3/ShapeShape#conv2d_transpose_2/BiasAdd:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :z]
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????z?*
paddingVALID*
strides
?
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z?Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV2#conv2d_transpose_1/BiasAdd:output:0#conv2d_transpose_3/BiasAdd:output:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????z?<?
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes

:<*
dtype0`
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          b
dense_3/Tensordot/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:a
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_3/Tensordot/transpose	Transposeconcatenate/concat:output:0!dense_3/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????z?<?
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????z??
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z?i
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????z?r
IdentityIdentitydense_3/Relu:activations:0^NoOp*
T0*0
_output_shapes
:?????????z??
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????z?: : : : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp:S O
0
_output_shapes
:?????????z?

_user_specified_namex
?
?
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_6469262

inputsC
(conv2d_transpose_readvariableop_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
stack/1Const*
_output_shapes
: *
dtype0*
value	B :zJ
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:?????????z?*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????z??
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????z
 
_user_specified_nameinputs
?
?
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_6469184

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
stack/1Const*
_output_shapes
: *
dtype0*
value	B :zI
stack/2Const*
_output_shapes
: *
dtype0*
value	B :I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????zg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????z?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_3_layer_call_fn_6469193

inputs"
unknown:?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_6467936?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
9__inference_source_separation_model_layer_call_fn_6468590
x"
unknown:?
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	?
	unknown_4:	?
	unknown_5:	?
	unknown_6:#
	unknown_7:
	unknown_8:$
	unknown_9:?

unknown_10:

unknown_11:	?

unknown_12:$

unknown_13:

unknown_14:%

unknown_15:?

unknown_16:

unknown_17:<

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_source_separation_model_layer_call_and_return_conditional_losses_6468252x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????z?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????z?: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:?????????z?

_user_specified_namex
?W
?
#__inference__traced_restore_6469468
file_prefixQ
6assignvariableop_source_separation_model_conv2d_kernel:?D
6assignvariableop_1_source_separation_model_conv2d_bias:T
:assignvariableop_2_source_separation_model_conv2d_1_kernel:F
8assignvariableop_3_source_separation_model_conv2d_1_bias:J
7assignvariableop_4_source_separation_model_dense_kernel:	?D
5assignvariableop_5_source_separation_model_dense_bias:	?L
9assignvariableop_6_source_separation_model_dense_1_kernel:	?E
7assignvariableop_7_source_separation_model_dense_1_bias:L
9assignvariableop_8_source_separation_model_dense_2_kernel:	?E
7assignvariableop_9_source_separation_model_dense_2_bias:]
Cassignvariableop_10_source_separation_model_conv2d_transpose_kernel:O
Aassignvariableop_11_source_separation_model_conv2d_transpose_bias:`
Eassignvariableop_12_source_separation_model_conv2d_transpose_1_kernel:?Q
Cassignvariableop_13_source_separation_model_conv2d_transpose_1_bias:_
Eassignvariableop_14_source_separation_model_conv2d_transpose_2_kernel:Q
Cassignvariableop_15_source_separation_model_conv2d_transpose_2_bias:`
Eassignvariableop_16_source_separation_model_conv2d_transpose_3_kernel:?Q
Cassignvariableop_17_source_separation_model_conv2d_transpose_3_bias:L
:assignvariableop_18_source_separation_model_dense_3_kernel:<F
8assignvariableop_19_source_separation_model_dense_3_bias:
identity_21??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB)dClass1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dClass1/bias/.ATTRIBUTES/VARIABLE_VALUEB)dClass2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dClass2/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv3Class1/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv3Class1/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv4Class1/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv4Class1/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv3Class2/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv3Class2/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv4Class2/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv4Class2/bias/.ATTRIBUTES/VARIABLE_VALUEB%out/kernel/.ATTRIBUTES/VARIABLE_VALUEB#out/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp6assignvariableop_source_separation_model_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp6assignvariableop_1_source_separation_model_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp:assignvariableop_2_source_separation_model_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp8assignvariableop_3_source_separation_model_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp7assignvariableop_4_source_separation_model_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp5assignvariableop_5_source_separation_model_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp9assignvariableop_6_source_separation_model_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp7assignvariableop_7_source_separation_model_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp9assignvariableop_8_source_separation_model_dense_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp7assignvariableop_9_source_separation_model_dense_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpCassignvariableop_10_source_separation_model_conv2d_transpose_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpAassignvariableop_11_source_separation_model_conv2d_transpose_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpEassignvariableop_12_source_separation_model_conv2d_transpose_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpCassignvariableop_13_source_separation_model_conv2d_transpose_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpEassignvariableop_14_source_separation_model_conv2d_transpose_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpCassignvariableop_15_source_separation_model_conv2d_transpose_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpEassignvariableop_16_source_separation_model_conv2d_transpose_3_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpCassignvariableop_17_source_separation_model_conv2d_transpose_3_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp:assignvariableop_18_source_separation_model_dense_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp8assignvariableop_19_source_separation_model_dense_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_21IdentityIdentity_20:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_21Identity_21:output:0*=
_input_shapes,
*: : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?#
?
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_6467888

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
D
input_19
serving_default_input_1:0?????????z?E
output_19
StatefulPartitionedCall:0?????????z?tensorflow/serving/predict:??
?
	conv1
	conv2
d1
dClass1
dClass2
conv3Class1
conv4Class1
conv3Class2
	conv4Class2


concat
out
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_model
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Akernel
Bbias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Kkernel
Lbias
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
0
1
2
3
4
5
#6
$7
)8
*9
/10
011
512
613
;14
<15
A16
B17
K18
L19"
trackable_list_wrapper
?
0
1
2
3
4
5
#6
$7
)8
*9
/10
011
512
613
;14
<15
A16
B17
K18
L19"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
@:>?2%source_separation_model/conv2d/kernel
1:/2#source_separation_model/conv2d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
A:?2'source_separation_model/conv2d_1/kernel
3:12%source_separation_model/conv2d_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
7:5	?2$source_separation_model/dense/kernel
1:/?2"source_separation_model/dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
 trainable_variables
!regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
9:7	?2&source_separation_model/dense_1/kernel
2:02$source_separation_model/dense_1/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
%	variables
&trainable_variables
'regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
9:7	?2&source_separation_model/dense_2/kernel
2:02$source_separation_model/dense_2/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
+	variables
,trainable_variables
-regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
I:G2/source_separation_model/conv2d_transpose/kernel
;:92-source_separation_model/conv2d_transpose/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
1	variables
2trainable_variables
3regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
L:J?21source_separation_model/conv2d_transpose_1/kernel
=:;2/source_separation_model/conv2d_transpose_1/bias
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
?
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
7	variables
8trainable_variables
9regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
K:I21source_separation_model/conv2d_transpose_2/kernel
=:;2/source_separation_model/conv2d_transpose_2/bias
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
=	variables
>trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
L:J?21source_separation_model/conv2d_transpose_3/kernel
=:;2/source_separation_model/conv2d_transpose_3/bias
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
~non_trainable_variables

layers
?metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
8:6<2&source_separation_model/dense_3/kernel
2:02$source_separation_model/dense_3/bias
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
9__inference_source_separation_model_layer_call_fn_6468295
9__inference_source_separation_model_layer_call_fn_6468590?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
T__inference_source_separation_model_layer_call_and_return_conditional_losses_6468792
T__inference_source_separation_model_layer_call_and_return_conditional_losses_6468498?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference__wrapped_model_6467751input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_layer_call_fn_6468801?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_layer_call_and_return_conditional_losses_6468811?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_1_layer_call_fn_6468820?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6468830?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_layer_call_fn_6468839?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_layer_call_and_return_conditional_losses_6468870?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_1_layer_call_fn_6468879?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_1_layer_call_and_return_conditional_losses_6468910?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_2_layer_call_fn_6468919?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_2_layer_call_and_return_conditional_losses_6468950?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_conv2d_transpose_layer_call_fn_6468959
2__inference_conv2d_transpose_layer_call_fn_6468968?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6469005
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6469028?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_conv2d_transpose_1_layer_call_fn_6469037
4__inference_conv2d_transpose_1_layer_call_fn_6469046?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_6469083
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_6469106?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_conv2d_transpose_2_layer_call_fn_6469115
4__inference_conv2d_transpose_2_layer_call_fn_6469124?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_6469161
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_6469184?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_conv2d_transpose_3_layer_call_fn_6469193
4__inference_conv2d_transpose_3_layer_call_fn_6469202?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_6469239
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_6469262?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_concatenate_layer_call_fn_6469268?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_concatenate_layer_call_and_return_conditional_losses_6469275?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_3_layer_call_fn_6469284?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_3_layer_call_and_return_conditional_losses_6469315?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_6468545input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_6467751?#$/056)*;<ABKL9?6
/?,
*?'
input_1?????????z?
? "<?9
7
output_1+?(
output_1?????????z??
H__inference_concatenate_layer_call_and_return_conditional_losses_6469275?l?i
b?_
]?Z
+?(
inputs/0?????????z?
+?(
inputs/1?????????z?
? ".?+
$?!
0?????????z?<
? ?
-__inference_concatenate_layer_call_fn_6469268?l?i
b?_
]?Z
+?(
inputs/0?????????z?
+?(
inputs/1?????????z?
? "!??????????z?<?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6468830l7?4
-?*
(?%
inputs?????????z
? "-?*
#? 
0?????????]
? ?
*__inference_conv2d_1_layer_call_fn_6468820_7?4
-?*
(?%
inputs?????????z
? " ??????????]?
C__inference_conv2d_layer_call_and_return_conditional_losses_6468811m8?5
.?+
)?&
inputs?????????z?
? "-?*
#? 
0?????????z
? ?
(__inference_conv2d_layer_call_fn_6468801`8?5
.?+
)?&
inputs?????????z?
? " ??????????z?
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_6469083?56I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_6469106m567?4
-?*
(?%
inputs?????????z
? ".?+
$?!
0?????????z?
? ?
4__inference_conv2d_transpose_1_layer_call_fn_6469037?56I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
4__inference_conv2d_transpose_1_layer_call_fn_6469046`567?4
-?*
(?%
inputs?????????z
? "!??????????z??
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_6469161?;<I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_6469184l;<7?4
-?*
(?%
inputs?????????]
? "-?*
#? 
0?????????z
? ?
4__inference_conv2d_transpose_2_layer_call_fn_6469115?;<I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
4__inference_conv2d_transpose_2_layer_call_fn_6469124_;<7?4
-?*
(?%
inputs?????????]
? " ??????????z?
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_6469239?ABI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_6469262mAB7?4
-?*
(?%
inputs?????????z
? ".?+
$?!
0?????????z?
? ?
4__inference_conv2d_transpose_3_layer_call_fn_6469193?ABI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
4__inference_conv2d_transpose_3_layer_call_fn_6469202`AB7?4
-?*
(?%
inputs?????????z
? "!??????????z??
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6469005?/0I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6469028l/07?4
-?*
(?%
inputs?????????]
? "-?*
#? 
0?????????z
? ?
2__inference_conv2d_transpose_layer_call_fn_6468959?/0I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
2__inference_conv2d_transpose_layer_call_fn_6468968_/07?4
-?*
(?%
inputs?????????]
? " ??????????z?
D__inference_dense_1_layer_call_and_return_conditional_losses_6468910m#$8?5
.?+
)?&
inputs?????????]?
? "-?*
#? 
0?????????]
? ?
)__inference_dense_1_layer_call_fn_6468879`#$8?5
.?+
)?&
inputs?????????]?
? " ??????????]?
D__inference_dense_2_layer_call_and_return_conditional_losses_6468950m)*8?5
.?+
)?&
inputs?????????]?
? "-?*
#? 
0?????????]
? ?
)__inference_dense_2_layer_call_fn_6468919`)*8?5
.?+
)?&
inputs?????????]?
? " ??????????]?
D__inference_dense_3_layer_call_and_return_conditional_losses_6469315nKL8?5
.?+
)?&
inputs?????????z?<
? ".?+
$?!
0?????????z?
? ?
)__inference_dense_3_layer_call_fn_6469284aKL8?5
.?+
)?&
inputs?????????z?<
? "!??????????z??
B__inference_dense_layer_call_and_return_conditional_losses_6468870m7?4
-?*
(?%
inputs?????????]
? ".?+
$?!
0?????????]?
? ?
'__inference_dense_layer_call_fn_6468839`7?4
-?*
(?%
inputs?????????]
? "!??????????]??
%__inference_signature_wrapper_6468545?#$/056)*;<ABKLD?A
? 
:?7
5
input_1*?'
input_1?????????z?"<?9
7
output_1+?(
output_1?????????z??
T__inference_source_separation_model_layer_call_and_return_conditional_losses_6468498?#$/056)*;<ABKL9?6
/?,
*?'
input_1?????????z?
? ".?+
$?!
0?????????z?
? ?
T__inference_source_separation_model_layer_call_and_return_conditional_losses_6468792{#$/056)*;<ABKL3?0
)?&
$?!
x?????????z?
? ".?+
$?!
0?????????z?
? ?
9__inference_source_separation_model_layer_call_fn_6468295t#$/056)*;<ABKL9?6
/?,
*?'
input_1?????????z?
? "!??????????z??
9__inference_source_separation_model_layer_call_fn_6468590n#$/056)*;<ABKL3?0
)?&
$?!
x?????????z?
? "!??????????z?