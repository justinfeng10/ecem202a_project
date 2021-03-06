??
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
 ?"serve*2.7.02v2.7.0-0-gc256c071bb28??
?
)source_separation_model_1/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)source_separation_model_1/conv2d_2/kernel
?
=source_separation_model_1/conv2d_2/kernel/Read/ReadVariableOpReadVariableOp)source_separation_model_1/conv2d_2/kernel*'
_output_shapes
:?*
dtype0
?
'source_separation_model_1/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'source_separation_model_1/conv2d_2/bias
?
;source_separation_model_1/conv2d_2/bias/Read/ReadVariableOpReadVariableOp'source_separation_model_1/conv2d_2/bias*
_output_shapes
:*
dtype0
?
)source_separation_model_1/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)source_separation_model_1/conv2d_3/kernel
?
=source_separation_model_1/conv2d_3/kernel/Read/ReadVariableOpReadVariableOp)source_separation_model_1/conv2d_3/kernel*&
_output_shapes
:*
dtype0
?
'source_separation_model_1/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'source_separation_model_1/conv2d_3/bias
?
;source_separation_model_1/conv2d_3/bias/Read/ReadVariableOpReadVariableOp'source_separation_model_1/conv2d_3/bias*
_output_shapes
:*
dtype0
?
(source_separation_model_1/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*9
shared_name*(source_separation_model_1/dense_6/kernel
?
<source_separation_model_1/dense_6/kernel/Read/ReadVariableOpReadVariableOp(source_separation_model_1/dense_6/kernel*
_output_shapes
:	?*
dtype0
?
&source_separation_model_1/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&source_separation_model_1/dense_6/bias
?
:source_separation_model_1/dense_6/bias/Read/ReadVariableOpReadVariableOp&source_separation_model_1/dense_6/bias*
_output_shapes	
:?*
dtype0
?
(source_separation_model_1/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*9
shared_name*(source_separation_model_1/dense_7/kernel
?
<source_separation_model_1/dense_7/kernel/Read/ReadVariableOpReadVariableOp(source_separation_model_1/dense_7/kernel*
_output_shapes
:	?*
dtype0
?
&source_separation_model_1/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&source_separation_model_1/dense_7/bias
?
:source_separation_model_1/dense_7/bias/Read/ReadVariableOpReadVariableOp&source_separation_model_1/dense_7/bias*
_output_shapes
:*
dtype0
?
(source_separation_model_1/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*9
shared_name*(source_separation_model_1/dense_8/kernel
?
<source_separation_model_1/dense_8/kernel/Read/ReadVariableOpReadVariableOp(source_separation_model_1/dense_8/kernel*
_output_shapes
:	?*
dtype0
?
&source_separation_model_1/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&source_separation_model_1/dense_8/bias
?
:source_separation_model_1/dense_8/bias/Read/ReadVariableOpReadVariableOp&source_separation_model_1/dense_8/bias*
_output_shapes
:*
dtype0
?
(source_separation_model_1/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*9
shared_name*(source_separation_model_1/dense_9/kernel
?
<source_separation_model_1/dense_9/kernel/Read/ReadVariableOpReadVariableOp(source_separation_model_1/dense_9/kernel*
_output_shapes
:	?*
dtype0
?
&source_separation_model_1/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&source_separation_model_1/dense_9/bias
?
:source_separation_model_1/dense_9/bias/Read/ReadVariableOpReadVariableOp&source_separation_model_1/dense_9/bias*
_output_shapes
:*
dtype0
?
)source_separation_model_1/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*:
shared_name+)source_separation_model_1/dense_10/kernel
?
=source_separation_model_1/dense_10/kernel/Read/ReadVariableOpReadVariableOp)source_separation_model_1/dense_10/kernel*
_output_shapes
:	?*
dtype0
?
'source_separation_model_1/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'source_separation_model_1/dense_10/bias
?
;source_separation_model_1/dense_10/bias/Read/ReadVariableOpReadVariableOp'source_separation_model_1/dense_10/bias*
_output_shapes
:*
dtype0
?
3source_separation_model_1/conv2d_transpose_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53source_separation_model_1/conv2d_transpose_8/kernel
?
Gsource_separation_model_1/conv2d_transpose_8/kernel/Read/ReadVariableOpReadVariableOp3source_separation_model_1/conv2d_transpose_8/kernel*&
_output_shapes
:*
dtype0
?
1source_separation_model_1/conv2d_transpose_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31source_separation_model_1/conv2d_transpose_8/bias
?
Esource_separation_model_1/conv2d_transpose_8/bias/Read/ReadVariableOpReadVariableOp1source_separation_model_1/conv2d_transpose_8/bias*
_output_shapes
:*
dtype0
?
3source_separation_model_1/conv2d_transpose_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*D
shared_name53source_separation_model_1/conv2d_transpose_9/kernel
?
Gsource_separation_model_1/conv2d_transpose_9/kernel/Read/ReadVariableOpReadVariableOp3source_separation_model_1/conv2d_transpose_9/kernel*'
_output_shapes
:?*
dtype0
?
1source_separation_model_1/conv2d_transpose_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31source_separation_model_1/conv2d_transpose_9/bias
?
Esource_separation_model_1/conv2d_transpose_9/bias/Read/ReadVariableOpReadVariableOp1source_separation_model_1/conv2d_transpose_9/bias*
_output_shapes
:*
dtype0
?
4source_separation_model_1/conv2d_transpose_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64source_separation_model_1/conv2d_transpose_10/kernel
?
Hsource_separation_model_1/conv2d_transpose_10/kernel/Read/ReadVariableOpReadVariableOp4source_separation_model_1/conv2d_transpose_10/kernel*&
_output_shapes
:*
dtype0
?
2source_separation_model_1/conv2d_transpose_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42source_separation_model_1/conv2d_transpose_10/bias
?
Fsource_separation_model_1/conv2d_transpose_10/bias/Read/ReadVariableOpReadVariableOp2source_separation_model_1/conv2d_transpose_10/bias*
_output_shapes
:*
dtype0
?
4source_separation_model_1/conv2d_transpose_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64source_separation_model_1/conv2d_transpose_11/kernel
?
Hsource_separation_model_1/conv2d_transpose_11/kernel/Read/ReadVariableOpReadVariableOp4source_separation_model_1/conv2d_transpose_11/kernel*'
_output_shapes
:?*
dtype0
?
2source_separation_model_1/conv2d_transpose_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42source_separation_model_1/conv2d_transpose_11/bias
?
Fsource_separation_model_1/conv2d_transpose_11/bias/Read/ReadVariableOpReadVariableOp2source_separation_model_1/conv2d_transpose_11/bias*
_output_shapes
:*
dtype0
?
4source_separation_model_1/conv2d_transpose_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64source_separation_model_1/conv2d_transpose_12/kernel
?
Hsource_separation_model_1/conv2d_transpose_12/kernel/Read/ReadVariableOpReadVariableOp4source_separation_model_1/conv2d_transpose_12/kernel*&
_output_shapes
:*
dtype0
?
2source_separation_model_1/conv2d_transpose_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42source_separation_model_1/conv2d_transpose_12/bias
?
Fsource_separation_model_1/conv2d_transpose_12/bias/Read/ReadVariableOpReadVariableOp2source_separation_model_1/conv2d_transpose_12/bias*
_output_shapes
:*
dtype0
?
4source_separation_model_1/conv2d_transpose_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64source_separation_model_1/conv2d_transpose_13/kernel
?
Hsource_separation_model_1/conv2d_transpose_13/kernel/Read/ReadVariableOpReadVariableOp4source_separation_model_1/conv2d_transpose_13/kernel*'
_output_shapes
:?*
dtype0
?
2source_separation_model_1/conv2d_transpose_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42source_separation_model_1/conv2d_transpose_13/bias
?
Fsource_separation_model_1/conv2d_transpose_13/bias/Read/ReadVariableOpReadVariableOp2source_separation_model_1/conv2d_transpose_13/bias*
_output_shapes
:*
dtype0
?
4source_separation_model_1/conv2d_transpose_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64source_separation_model_1/conv2d_transpose_14/kernel
?
Hsource_separation_model_1/conv2d_transpose_14/kernel/Read/ReadVariableOpReadVariableOp4source_separation_model_1/conv2d_transpose_14/kernel*&
_output_shapes
:*
dtype0
?
2source_separation_model_1/conv2d_transpose_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42source_separation_model_1/conv2d_transpose_14/bias
?
Fsource_separation_model_1/conv2d_transpose_14/bias/Read/ReadVariableOpReadVariableOp2source_separation_model_1/conv2d_transpose_14/bias*
_output_shapes
:*
dtype0
?
4source_separation_model_1/conv2d_transpose_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64source_separation_model_1/conv2d_transpose_15/kernel
?
Hsource_separation_model_1/conv2d_transpose_15/kernel/Read/ReadVariableOpReadVariableOp4source_separation_model_1/conv2d_transpose_15/kernel*'
_output_shapes
:?*
dtype0
?
2source_separation_model_1/conv2d_transpose_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42source_separation_model_1/conv2d_transpose_15/bias
?
Fsource_separation_model_1/conv2d_transpose_15/bias/Read/ReadVariableOpReadVariableOp2source_separation_model_1/conv2d_transpose_15/bias*
_output_shapes
:*
dtype0
?
)source_separation_model_1/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*:
shared_name+)source_separation_model_1/dense_11/kernel
?
=source_separation_model_1/dense_11/kernel/Read/ReadVariableOpReadVariableOp)source_separation_model_1/dense_11/kernel*
_output_shapes

:x*
dtype0
?
'source_separation_model_1/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'source_separation_model_1/dense_11/bias
?
;source_separation_model_1/dense_11/bias/Read/ReadVariableOpReadVariableOp'source_separation_model_1/dense_11/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?P
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?P
value?PB?P B?P
?
	conv1
	conv2
d1
dClass0
dClass1
dClass2
dClass3
conv3Class0
	conv4Class0

conv3Class1
conv4Class1
conv3Class2
conv4Class2
conv3Class3
conv4Class3

concat
out
	variables
trainable_variables
regularization_losses
	keras_api

signatures
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
h

Gkernel
Hbias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
h

Mkernel
Nbias
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
h

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
h

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
h

_kernel
`bias
a	variables
btrainable_variables
cregularization_losses
d	keras_api
h

ekernel
fbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
h

kkernel
lbias
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
R
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
h

ukernel
vbias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
?
0
1
2
3
#4
$5
)6
*7
/8
09
510
611
;12
<13
A14
B15
G16
H17
M18
N19
S20
T21
Y22
Z23
_24
`25
e26
f27
k28
l29
u30
v31
?
0
1
2
3
#4
$5
)6
*7
/8
09
510
611
;12
<13
A14
B15
G16
H17
M18
N19
S20
T21
Y22
Z23
_24
`25
e26
f27
k28
l29
u30
v31
 
?
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 
fd
VARIABLE_VALUE)source_separation_model_1/conv2d_2/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE'source_separation_model_1/conv2d_2/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
fd
VARIABLE_VALUE)source_separation_model_1/conv2d_3/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE'source_separation_model_1/conv2d_3/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
 trainable_variables
!regularization_losses
b`
VARIABLE_VALUE(source_separation_model_1/dense_6/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE&source_separation_model_1/dense_6/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1

#0
$1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
%	variables
&trainable_variables
'regularization_losses
ge
VARIABLE_VALUE(source_separation_model_1/dense_7/kernel)dClass0/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&source_separation_model_1/dense_7/bias'dClass0/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1

)0
*1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
+	variables
,trainable_variables
-regularization_losses
ge
VARIABLE_VALUE(source_separation_model_1/dense_8/kernel)dClass1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&source_separation_model_1/dense_8/bias'dClass1/bias/.ATTRIBUTES/VARIABLE_VALUE

/0
01

/0
01
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
1	variables
2trainable_variables
3regularization_losses
ge
VARIABLE_VALUE(source_separation_model_1/dense_9/kernel)dClass2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&source_separation_model_1/dense_9/bias'dClass2/bias/.ATTRIBUTES/VARIABLE_VALUE

50
61

50
61
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
7	variables
8trainable_variables
9regularization_losses
hf
VARIABLE_VALUE)source_separation_model_1/dense_10/kernel)dClass3/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'source_separation_model_1/dense_10/bias'dClass3/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1

;0
<1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
=	variables
>trainable_variables
?regularization_losses
vt
VARIABLE_VALUE3source_separation_model_1/conv2d_transpose_8/kernel-conv3Class0/kernel/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE1source_separation_model_1/conv2d_transpose_8/bias+conv3Class0/bias/.ATTRIBUTES/VARIABLE_VALUE

A0
B1

A0
B1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
vt
VARIABLE_VALUE3source_separation_model_1/conv2d_transpose_9/kernel-conv4Class0/kernel/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE1source_separation_model_1/conv2d_transpose_9/bias+conv4Class0/bias/.ATTRIBUTES/VARIABLE_VALUE

G0
H1

G0
H1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
wu
VARIABLE_VALUE4source_separation_model_1/conv2d_transpose_10/kernel-conv3Class1/kernel/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE2source_separation_model_1/conv2d_transpose_10/bias+conv3Class1/bias/.ATTRIBUTES/VARIABLE_VALUE

M0
N1

M0
N1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
wu
VARIABLE_VALUE4source_separation_model_1/conv2d_transpose_11/kernel-conv4Class1/kernel/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE2source_separation_model_1/conv2d_transpose_11/bias+conv4Class1/bias/.ATTRIBUTES/VARIABLE_VALUE

S0
T1

S0
T1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
wu
VARIABLE_VALUE4source_separation_model_1/conv2d_transpose_12/kernel-conv3Class2/kernel/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE2source_separation_model_1/conv2d_transpose_12/bias+conv3Class2/bias/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1

Y0
Z1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
[	variables
\trainable_variables
]regularization_losses
wu
VARIABLE_VALUE4source_separation_model_1/conv2d_transpose_13/kernel-conv4Class2/kernel/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE2source_separation_model_1/conv2d_transpose_13/bias+conv4Class2/bias/.ATTRIBUTES/VARIABLE_VALUE

_0
`1

_0
`1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
a	variables
btrainable_variables
cregularization_losses
wu
VARIABLE_VALUE4source_separation_model_1/conv2d_transpose_14/kernel-conv3Class3/kernel/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE2source_separation_model_1/conv2d_transpose_14/bias+conv3Class3/bias/.ATTRIBUTES/VARIABLE_VALUE

e0
f1

e0
f1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
g	variables
htrainable_variables
iregularization_losses
wu
VARIABLE_VALUE4source_separation_model_1/conv2d_transpose_15/kernel-conv4Class3/kernel/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE2source_separation_model_1/conv2d_transpose_15/bias+conv4Class3/bias/.ATTRIBUTES/VARIABLE_VALUE

k0
l1

k0
l1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
q	variables
rtrainable_variables
sregularization_losses
db
VARIABLE_VALUE)source_separation_model_1/dense_11/kernel%out/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE'source_separation_model_1/dense_11/bias#out/bias/.ATTRIBUTES/VARIABLE_VALUE

u0
v1

u0
v1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
w	variables
xtrainable_variables
yregularization_losses
 
~
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
11
12
13
14
15
16
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
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1)source_separation_model_1/conv2d_2/kernel'source_separation_model_1/conv2d_2/bias)source_separation_model_1/conv2d_3/kernel'source_separation_model_1/conv2d_3/bias(source_separation_model_1/dense_6/kernel&source_separation_model_1/dense_6/bias(source_separation_model_1/dense_7/kernel&source_separation_model_1/dense_7/bias3source_separation_model_1/conv2d_transpose_8/kernel1source_separation_model_1/conv2d_transpose_8/bias3source_separation_model_1/conv2d_transpose_9/kernel1source_separation_model_1/conv2d_transpose_9/bias(source_separation_model_1/dense_8/kernel&source_separation_model_1/dense_8/bias4source_separation_model_1/conv2d_transpose_10/kernel2source_separation_model_1/conv2d_transpose_10/bias4source_separation_model_1/conv2d_transpose_11/kernel2source_separation_model_1/conv2d_transpose_11/bias(source_separation_model_1/dense_9/kernel&source_separation_model_1/dense_9/bias4source_separation_model_1/conv2d_transpose_12/kernel2source_separation_model_1/conv2d_transpose_12/bias4source_separation_model_1/conv2d_transpose_13/kernel2source_separation_model_1/conv2d_transpose_13/bias)source_separation_model_1/dense_10/kernel'source_separation_model_1/dense_10/bias4source_separation_model_1/conv2d_transpose_14/kernel2source_separation_model_1/conv2d_transpose_14/bias4source_separation_model_1/conv2d_transpose_15/kernel2source_separation_model_1/conv2d_transpose_15/bias)source_separation_model_1/dense_11/kernel'source_separation_model_1/dense_11/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_21061078
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename=source_separation_model_1/conv2d_2/kernel/Read/ReadVariableOp;source_separation_model_1/conv2d_2/bias/Read/ReadVariableOp=source_separation_model_1/conv2d_3/kernel/Read/ReadVariableOp;source_separation_model_1/conv2d_3/bias/Read/ReadVariableOp<source_separation_model_1/dense_6/kernel/Read/ReadVariableOp:source_separation_model_1/dense_6/bias/Read/ReadVariableOp<source_separation_model_1/dense_7/kernel/Read/ReadVariableOp:source_separation_model_1/dense_7/bias/Read/ReadVariableOp<source_separation_model_1/dense_8/kernel/Read/ReadVariableOp:source_separation_model_1/dense_8/bias/Read/ReadVariableOp<source_separation_model_1/dense_9/kernel/Read/ReadVariableOp:source_separation_model_1/dense_9/bias/Read/ReadVariableOp=source_separation_model_1/dense_10/kernel/Read/ReadVariableOp;source_separation_model_1/dense_10/bias/Read/ReadVariableOpGsource_separation_model_1/conv2d_transpose_8/kernel/Read/ReadVariableOpEsource_separation_model_1/conv2d_transpose_8/bias/Read/ReadVariableOpGsource_separation_model_1/conv2d_transpose_9/kernel/Read/ReadVariableOpEsource_separation_model_1/conv2d_transpose_9/bias/Read/ReadVariableOpHsource_separation_model_1/conv2d_transpose_10/kernel/Read/ReadVariableOpFsource_separation_model_1/conv2d_transpose_10/bias/Read/ReadVariableOpHsource_separation_model_1/conv2d_transpose_11/kernel/Read/ReadVariableOpFsource_separation_model_1/conv2d_transpose_11/bias/Read/ReadVariableOpHsource_separation_model_1/conv2d_transpose_12/kernel/Read/ReadVariableOpFsource_separation_model_1/conv2d_transpose_12/bias/Read/ReadVariableOpHsource_separation_model_1/conv2d_transpose_13/kernel/Read/ReadVariableOpFsource_separation_model_1/conv2d_transpose_13/bias/Read/ReadVariableOpHsource_separation_model_1/conv2d_transpose_14/kernel/Read/ReadVariableOpFsource_separation_model_1/conv2d_transpose_14/bias/Read/ReadVariableOpHsource_separation_model_1/conv2d_transpose_15/kernel/Read/ReadVariableOpFsource_separation_model_1/conv2d_transpose_15/bias/Read/ReadVariableOp=source_separation_model_1/dense_11/kernel/Read/ReadVariableOp;source_separation_model_1/dense_11/bias/Read/ReadVariableOpConst*-
Tin&
$2"*
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
GPU 2J 8? **
f%R#
!__inference__traced_save_21062517
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename)source_separation_model_1/conv2d_2/kernel'source_separation_model_1/conv2d_2/bias)source_separation_model_1/conv2d_3/kernel'source_separation_model_1/conv2d_3/bias(source_separation_model_1/dense_6/kernel&source_separation_model_1/dense_6/bias(source_separation_model_1/dense_7/kernel&source_separation_model_1/dense_7/bias(source_separation_model_1/dense_8/kernel&source_separation_model_1/dense_8/bias(source_separation_model_1/dense_9/kernel&source_separation_model_1/dense_9/bias)source_separation_model_1/dense_10/kernel'source_separation_model_1/dense_10/bias3source_separation_model_1/conv2d_transpose_8/kernel1source_separation_model_1/conv2d_transpose_8/bias3source_separation_model_1/conv2d_transpose_9/kernel1source_separation_model_1/conv2d_transpose_9/bias4source_separation_model_1/conv2d_transpose_10/kernel2source_separation_model_1/conv2d_transpose_10/bias4source_separation_model_1/conv2d_transpose_11/kernel2source_separation_model_1/conv2d_transpose_11/bias4source_separation_model_1/conv2d_transpose_12/kernel2source_separation_model_1/conv2d_transpose_12/bias4source_separation_model_1/conv2d_transpose_13/kernel2source_separation_model_1/conv2d_transpose_13/bias4source_separation_model_1/conv2d_transpose_14/kernel2source_separation_model_1/conv2d_transpose_14/bias4source_separation_model_1/conv2d_transpose_15/kernel2source_separation_model_1/conv2d_transpose_15/bias)source_separation_model_1/dense_11/kernel'source_separation_model_1/dense_11/bias*,
Tin%
#2!*
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
GPU 2J 8? *-
f(R&
$__inference__traced_restore_21062623??
?
?
P__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_21061795

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
?
?
E__inference_dense_9_layer_call_and_return_conditional_losses_21060421

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
?^
?
W__inference_source_separation_model_1_layer_call_and_return_conditional_losses_21060625
x,
conv2d_2_21060146:?
conv2d_2_21060148:+
conv2d_3_21060162:
conv2d_3_21060164:#
dense_6_21060199:	?
dense_6_21060201:	?#
dense_7_21060236:	?
dense_7_21060238:5
conv2d_transpose_8_21060264:)
conv2d_transpose_8_21060266:6
conv2d_transpose_9_21060292:?)
conv2d_transpose_9_21060294:#
dense_8_21060329:	?
dense_8_21060331:6
conv2d_transpose_10_21060357:*
conv2d_transpose_10_21060359:7
conv2d_transpose_11_21060385:?*
conv2d_transpose_11_21060387:#
dense_9_21060422:	?
dense_9_21060424:6
conv2d_transpose_12_21060450:*
conv2d_transpose_12_21060452:7
conv2d_transpose_13_21060478:?*
conv2d_transpose_13_21060480:$
dense_10_21060515:	?
dense_10_21060517:6
conv2d_transpose_14_21060543:*
conv2d_transpose_14_21060545:7
conv2d_transpose_15_21060571:?*
conv2d_transpose_15_21060573:#
dense_11_21060619:x
dense_11_21060621:
identity?? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?+conv2d_transpose_10/StatefulPartitionedCall?+conv2d_transpose_11/StatefulPartitionedCall?+conv2d_transpose_12/StatefulPartitionedCall?+conv2d_transpose_13/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?+conv2d_transpose_15/StatefulPartitionedCall?*conv2d_transpose_8/StatefulPartitionedCall?*conv2d_transpose_9/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallxconv2d_2_21060146conv2d_2_21060148*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_21060145?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_21060162conv2d_3_21060164*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_21060161?
dense_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0dense_6_21060199dense_6_21060201*
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
GPU 2J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_21060198?
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_21060236dense_7_21060238*
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
E__inference_dense_7_layer_call_and_return_conditional_losses_21060235?
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0conv2d_transpose_8_21060264conv2d_transpose_8_21060266*
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
GPU 2J 8? *Y
fTRR
P__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_21060263?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0conv2d_transpose_9_21060292conv2d_transpose_9_21060294*
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
GPU 2J 8? *Y
fTRR
P__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_21060291?
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_8_21060329dense_8_21060331*
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
E__inference_dense_8_layer_call_and_return_conditional_losses_21060328?
+conv2d_transpose_10/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0conv2d_transpose_10_21060357conv2d_transpose_10_21060359*
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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_21060356?
+conv2d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_10/StatefulPartitionedCall:output:0conv2d_transpose_11_21060385conv2d_transpose_11_21060387*
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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_21060384?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_9_21060422dense_9_21060424*
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
E__inference_dense_9_layer_call_and_return_conditional_losses_21060421?
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0conv2d_transpose_12_21060450conv2d_transpose_12_21060452*
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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_21060449?
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0conv2d_transpose_13_21060478conv2d_transpose_13_21060480*
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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_21060477?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_10_21060515dense_10_21060517*
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
GPU 2J 8? *O
fJRH
F__inference_dense_10_layer_call_and_return_conditional_losses_21060514?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0conv2d_transpose_14_21060543conv2d_transpose_14_21060545*
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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_21060542?
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0conv2d_transpose_15_21060571conv2d_transpose_15_21060573*
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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_21060570?
concatenate_1/PartitionedCallPartitionedCall3conv2d_transpose_9/StatefulPartitionedCall:output:04conv2d_transpose_11/StatefulPartitionedCall:output:04conv2d_transpose_13/StatefulPartitionedCall:output:04conv2d_transpose_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?x* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concatenate_1_layer_call_and_return_conditional_losses_21060585?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_11_21060619dense_11_21060621*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_21060618?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????z??
NoOpNoOp!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall,^conv2d_transpose_10/StatefulPartitionedCall,^conv2d_transpose_11/StatefulPartitionedCall,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall+^conv2d_transpose_8/StatefulPartitionedCall+^conv2d_transpose_9/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:?????????z?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2Z
+conv2d_transpose_10/StatefulPartitionedCall+conv2d_transpose_10/StatefulPartitionedCall2Z
+conv2d_transpose_11/StatefulPartitionedCall+conv2d_transpose_11/StatefulPartitionedCall2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:S O
0
_output_shapes
:?????????z?

_user_specified_namex
?
?
6__inference_conv2d_transpose_11_layer_call_fn_21061969

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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_21060384x
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
?
?
K__inference_concatenate_1_layer_call_and_return_conditional_losses_21060585

inputs
inputs_1
inputs_2
inputs_3
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputsinputs_1inputs_2inputs_3concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????z?x`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????z?x"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesr
p:?????????z?:?????????z?:?????????z?:?????????z?:X T
0
_output_shapes
:?????????z?
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????z?
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????z?
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????z?
 
_user_specified_nameinputs
?
?
Q__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_21062029

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
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_21060570

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
??
?)
#__inference__wrapped_model_21059744
input_1\
Asource_separation_model_1_conv2d_2_conv2d_readvariableop_resource:?P
Bsource_separation_model_1_conv2d_2_biasadd_readvariableop_resource:[
Asource_separation_model_1_conv2d_3_conv2d_readvariableop_resource:P
Bsource_separation_model_1_conv2d_3_biasadd_readvariableop_resource:V
Csource_separation_model_1_dense_6_tensordot_readvariableop_resource:	?P
Asource_separation_model_1_dense_6_biasadd_readvariableop_resource:	?V
Csource_separation_model_1_dense_7_tensordot_readvariableop_resource:	?O
Asource_separation_model_1_dense_7_biasadd_readvariableop_resource:o
Usource_separation_model_1_conv2d_transpose_8_conv2d_transpose_readvariableop_resource:Z
Lsource_separation_model_1_conv2d_transpose_8_biasadd_readvariableop_resource:p
Usource_separation_model_1_conv2d_transpose_9_conv2d_transpose_readvariableop_resource:?Z
Lsource_separation_model_1_conv2d_transpose_9_biasadd_readvariableop_resource:V
Csource_separation_model_1_dense_8_tensordot_readvariableop_resource:	?O
Asource_separation_model_1_dense_8_biasadd_readvariableop_resource:p
Vsource_separation_model_1_conv2d_transpose_10_conv2d_transpose_readvariableop_resource:[
Msource_separation_model_1_conv2d_transpose_10_biasadd_readvariableop_resource:q
Vsource_separation_model_1_conv2d_transpose_11_conv2d_transpose_readvariableop_resource:?[
Msource_separation_model_1_conv2d_transpose_11_biasadd_readvariableop_resource:V
Csource_separation_model_1_dense_9_tensordot_readvariableop_resource:	?O
Asource_separation_model_1_dense_9_biasadd_readvariableop_resource:p
Vsource_separation_model_1_conv2d_transpose_12_conv2d_transpose_readvariableop_resource:[
Msource_separation_model_1_conv2d_transpose_12_biasadd_readvariableop_resource:q
Vsource_separation_model_1_conv2d_transpose_13_conv2d_transpose_readvariableop_resource:?[
Msource_separation_model_1_conv2d_transpose_13_biasadd_readvariableop_resource:W
Dsource_separation_model_1_dense_10_tensordot_readvariableop_resource:	?P
Bsource_separation_model_1_dense_10_biasadd_readvariableop_resource:p
Vsource_separation_model_1_conv2d_transpose_14_conv2d_transpose_readvariableop_resource:[
Msource_separation_model_1_conv2d_transpose_14_biasadd_readvariableop_resource:q
Vsource_separation_model_1_conv2d_transpose_15_conv2d_transpose_readvariableop_resource:?[
Msource_separation_model_1_conv2d_transpose_15_biasadd_readvariableop_resource:V
Dsource_separation_model_1_dense_11_tensordot_readvariableop_resource:xP
Bsource_separation_model_1_dense_11_biasadd_readvariableop_resource:
identity??9source_separation_model_1/conv2d_2/BiasAdd/ReadVariableOp?8source_separation_model_1/conv2d_2/Conv2D/ReadVariableOp?9source_separation_model_1/conv2d_3/BiasAdd/ReadVariableOp?8source_separation_model_1/conv2d_3/Conv2D/ReadVariableOp?Dsource_separation_model_1/conv2d_transpose_10/BiasAdd/ReadVariableOp?Msource_separation_model_1/conv2d_transpose_10/conv2d_transpose/ReadVariableOp?Dsource_separation_model_1/conv2d_transpose_11/BiasAdd/ReadVariableOp?Msource_separation_model_1/conv2d_transpose_11/conv2d_transpose/ReadVariableOp?Dsource_separation_model_1/conv2d_transpose_12/BiasAdd/ReadVariableOp?Msource_separation_model_1/conv2d_transpose_12/conv2d_transpose/ReadVariableOp?Dsource_separation_model_1/conv2d_transpose_13/BiasAdd/ReadVariableOp?Msource_separation_model_1/conv2d_transpose_13/conv2d_transpose/ReadVariableOp?Dsource_separation_model_1/conv2d_transpose_14/BiasAdd/ReadVariableOp?Msource_separation_model_1/conv2d_transpose_14/conv2d_transpose/ReadVariableOp?Dsource_separation_model_1/conv2d_transpose_15/BiasAdd/ReadVariableOp?Msource_separation_model_1/conv2d_transpose_15/conv2d_transpose/ReadVariableOp?Csource_separation_model_1/conv2d_transpose_8/BiasAdd/ReadVariableOp?Lsource_separation_model_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOp?Csource_separation_model_1/conv2d_transpose_9/BiasAdd/ReadVariableOp?Lsource_separation_model_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOp?9source_separation_model_1/dense_10/BiasAdd/ReadVariableOp?;source_separation_model_1/dense_10/Tensordot/ReadVariableOp?9source_separation_model_1/dense_11/BiasAdd/ReadVariableOp?;source_separation_model_1/dense_11/Tensordot/ReadVariableOp?8source_separation_model_1/dense_6/BiasAdd/ReadVariableOp?:source_separation_model_1/dense_6/Tensordot/ReadVariableOp?8source_separation_model_1/dense_7/BiasAdd/ReadVariableOp?:source_separation_model_1/dense_7/Tensordot/ReadVariableOp?8source_separation_model_1/dense_8/BiasAdd/ReadVariableOp?:source_separation_model_1/dense_8/Tensordot/ReadVariableOp?8source_separation_model_1/dense_9/BiasAdd/ReadVariableOp?:source_separation_model_1/dense_9/Tensordot/ReadVariableOp?
8source_separation_model_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOpAsource_separation_model_1_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
)source_separation_model_1/conv2d_2/Conv2DConv2Dinput_1@source_separation_model_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
?
9source_separation_model_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpBsource_separation_model_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
*source_separation_model_1/conv2d_2/BiasAddBiasAdd2source_separation_model_1/conv2d_2/Conv2D:output:0Asource_separation_model_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z?
8source_separation_model_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOpAsource_separation_model_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
)source_separation_model_1/conv2d_3/Conv2DConv2D3source_separation_model_1/conv2d_2/BiasAdd:output:0@source_separation_model_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]*
paddingVALID*
strides
?
9source_separation_model_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpBsource_separation_model_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
*source_separation_model_1/conv2d_3/BiasAddBiasAdd2source_separation_model_1/conv2d_3/Conv2D:output:0Asource_separation_model_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]?
:source_separation_model_1/dense_6/Tensordot/ReadVariableOpReadVariableOpCsource_separation_model_1_dense_6_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0z
0source_separation_model_1/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
0source_separation_model_1/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          ?
1source_separation_model_1/dense_6/Tensordot/ShapeShape3source_separation_model_1/conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:{
9source_separation_model_1/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4source_separation_model_1/dense_6/Tensordot/GatherV2GatherV2:source_separation_model_1/dense_6/Tensordot/Shape:output:09source_separation_model_1/dense_6/Tensordot/free:output:0Bsource_separation_model_1/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
;source_separation_model_1/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6source_separation_model_1/dense_6/Tensordot/GatherV2_1GatherV2:source_separation_model_1/dense_6/Tensordot/Shape:output:09source_separation_model_1/dense_6/Tensordot/axes:output:0Dsource_separation_model_1/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
1source_separation_model_1/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
0source_separation_model_1/dense_6/Tensordot/ProdProd=source_separation_model_1/dense_6/Tensordot/GatherV2:output:0:source_separation_model_1/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3source_separation_model_1/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
2source_separation_model_1/dense_6/Tensordot/Prod_1Prod?source_separation_model_1/dense_6/Tensordot/GatherV2_1:output:0<source_separation_model_1/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7source_separation_model_1/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2source_separation_model_1/dense_6/Tensordot/concatConcatV29source_separation_model_1/dense_6/Tensordot/free:output:09source_separation_model_1/dense_6/Tensordot/axes:output:0@source_separation_model_1/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1source_separation_model_1/dense_6/Tensordot/stackPack9source_separation_model_1/dense_6/Tensordot/Prod:output:0;source_separation_model_1/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
5source_separation_model_1/dense_6/Tensordot/transpose	Transpose3source_separation_model_1/conv2d_3/BiasAdd:output:0;source_separation_model_1/dense_6/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????]?
3source_separation_model_1/dense_6/Tensordot/ReshapeReshape9source_separation_model_1/dense_6/Tensordot/transpose:y:0:source_separation_model_1/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
2source_separation_model_1/dense_6/Tensordot/MatMulMatMul<source_separation_model_1/dense_6/Tensordot/Reshape:output:0Bsource_separation_model_1/dense_6/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????~
3source_separation_model_1/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?{
9source_separation_model_1/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4source_separation_model_1/dense_6/Tensordot/concat_1ConcatV2=source_separation_model_1/dense_6/Tensordot/GatherV2:output:0<source_separation_model_1/dense_6/Tensordot/Const_2:output:0Bsource_separation_model_1/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
+source_separation_model_1/dense_6/TensordotReshape<source_separation_model_1/dense_6/Tensordot/MatMul:product:0=source_separation_model_1/dense_6/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????]??
8source_separation_model_1/dense_6/BiasAdd/ReadVariableOpReadVariableOpAsource_separation_model_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
)source_separation_model_1/dense_6/BiasAddBiasAdd4source_separation_model_1/dense_6/Tensordot:output:0@source_separation_model_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????]??
&source_separation_model_1/dense_6/ReluRelu2source_separation_model_1/dense_6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????]??
:source_separation_model_1/dense_7/Tensordot/ReadVariableOpReadVariableOpCsource_separation_model_1_dense_7_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0z
0source_separation_model_1/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
0source_separation_model_1/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          ?
1source_separation_model_1/dense_7/Tensordot/ShapeShape4source_separation_model_1/dense_6/Relu:activations:0*
T0*
_output_shapes
:{
9source_separation_model_1/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4source_separation_model_1/dense_7/Tensordot/GatherV2GatherV2:source_separation_model_1/dense_7/Tensordot/Shape:output:09source_separation_model_1/dense_7/Tensordot/free:output:0Bsource_separation_model_1/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
;source_separation_model_1/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6source_separation_model_1/dense_7/Tensordot/GatherV2_1GatherV2:source_separation_model_1/dense_7/Tensordot/Shape:output:09source_separation_model_1/dense_7/Tensordot/axes:output:0Dsource_separation_model_1/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
1source_separation_model_1/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
0source_separation_model_1/dense_7/Tensordot/ProdProd=source_separation_model_1/dense_7/Tensordot/GatherV2:output:0:source_separation_model_1/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3source_separation_model_1/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
2source_separation_model_1/dense_7/Tensordot/Prod_1Prod?source_separation_model_1/dense_7/Tensordot/GatherV2_1:output:0<source_separation_model_1/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7source_separation_model_1/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2source_separation_model_1/dense_7/Tensordot/concatConcatV29source_separation_model_1/dense_7/Tensordot/free:output:09source_separation_model_1/dense_7/Tensordot/axes:output:0@source_separation_model_1/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1source_separation_model_1/dense_7/Tensordot/stackPack9source_separation_model_1/dense_7/Tensordot/Prod:output:0;source_separation_model_1/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
5source_separation_model_1/dense_7/Tensordot/transpose	Transpose4source_separation_model_1/dense_6/Relu:activations:0;source_separation_model_1/dense_7/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????]??
3source_separation_model_1/dense_7/Tensordot/ReshapeReshape9source_separation_model_1/dense_7/Tensordot/transpose:y:0:source_separation_model_1/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
2source_separation_model_1/dense_7/Tensordot/MatMulMatMul<source_separation_model_1/dense_7/Tensordot/Reshape:output:0Bsource_separation_model_1/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????}
3source_separation_model_1/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:{
9source_separation_model_1/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4source_separation_model_1/dense_7/Tensordot/concat_1ConcatV2=source_separation_model_1/dense_7/Tensordot/GatherV2:output:0<source_separation_model_1/dense_7/Tensordot/Const_2:output:0Bsource_separation_model_1/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
+source_separation_model_1/dense_7/TensordotReshape<source_separation_model_1/dense_7/Tensordot/MatMul:product:0=source_separation_model_1/dense_7/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????]?
8source_separation_model_1/dense_7/BiasAdd/ReadVariableOpReadVariableOpAsource_separation_model_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
)source_separation_model_1/dense_7/BiasAddBiasAdd4source_separation_model_1/dense_7/Tensordot:output:0@source_separation_model_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]?
&source_separation_model_1/dense_7/ReluRelu2source_separation_model_1/dense_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????]?
2source_separation_model_1/conv2d_transpose_8/ShapeShape4source_separation_model_1/dense_7/Relu:activations:0*
T0*
_output_shapes
:?
@source_separation_model_1/conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bsource_separation_model_1/conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bsource_separation_model_1/conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:source_separation_model_1/conv2d_transpose_8/strided_sliceStridedSlice;source_separation_model_1/conv2d_transpose_8/Shape:output:0Isource_separation_model_1/conv2d_transpose_8/strided_slice/stack:output:0Ksource_separation_model_1/conv2d_transpose_8/strided_slice/stack_1:output:0Ksource_separation_model_1/conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4source_separation_model_1/conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :zv
4source_separation_model_1/conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :v
4source_separation_model_1/conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
2source_separation_model_1/conv2d_transpose_8/stackPackCsource_separation_model_1/conv2d_transpose_8/strided_slice:output:0=source_separation_model_1/conv2d_transpose_8/stack/1:output:0=source_separation_model_1/conv2d_transpose_8/stack/2:output:0=source_separation_model_1/conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:?
Bsource_separation_model_1/conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Dsource_separation_model_1/conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Dsource_separation_model_1/conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
<source_separation_model_1/conv2d_transpose_8/strided_slice_1StridedSlice;source_separation_model_1/conv2d_transpose_8/stack:output:0Ksource_separation_model_1/conv2d_transpose_8/strided_slice_1/stack:output:0Msource_separation_model_1/conv2d_transpose_8/strided_slice_1/stack_1:output:0Msource_separation_model_1/conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Lsource_separation_model_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOpUsource_separation_model_1_conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
=source_separation_model_1/conv2d_transpose_8/conv2d_transposeConv2DBackpropInput;source_separation_model_1/conv2d_transpose_8/stack:output:0Tsource_separation_model_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:04source_separation_model_1/dense_7/Relu:activations:0*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
?
Csource_separation_model_1/conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOpLsource_separation_model_1_conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
4source_separation_model_1/conv2d_transpose_8/BiasAddBiasAddFsource_separation_model_1/conv2d_transpose_8/conv2d_transpose:output:0Ksource_separation_model_1/conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z?
2source_separation_model_1/conv2d_transpose_9/ShapeShape=source_separation_model_1/conv2d_transpose_8/BiasAdd:output:0*
T0*
_output_shapes
:?
@source_separation_model_1/conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bsource_separation_model_1/conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bsource_separation_model_1/conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:source_separation_model_1/conv2d_transpose_9/strided_sliceStridedSlice;source_separation_model_1/conv2d_transpose_9/Shape:output:0Isource_separation_model_1/conv2d_transpose_9/strided_slice/stack:output:0Ksource_separation_model_1/conv2d_transpose_9/strided_slice/stack_1:output:0Ksource_separation_model_1/conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4source_separation_model_1/conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :zw
4source_separation_model_1/conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?v
4source_separation_model_1/conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
2source_separation_model_1/conv2d_transpose_9/stackPackCsource_separation_model_1/conv2d_transpose_9/strided_slice:output:0=source_separation_model_1/conv2d_transpose_9/stack/1:output:0=source_separation_model_1/conv2d_transpose_9/stack/2:output:0=source_separation_model_1/conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:?
Bsource_separation_model_1/conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Dsource_separation_model_1/conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Dsource_separation_model_1/conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
<source_separation_model_1/conv2d_transpose_9/strided_slice_1StridedSlice;source_separation_model_1/conv2d_transpose_9/stack:output:0Ksource_separation_model_1/conv2d_transpose_9/strided_slice_1/stack:output:0Msource_separation_model_1/conv2d_transpose_9/strided_slice_1/stack_1:output:0Msource_separation_model_1/conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Lsource_separation_model_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOpUsource_separation_model_1_conv2d_transpose_9_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
=source_separation_model_1/conv2d_transpose_9/conv2d_transposeConv2DBackpropInput;source_separation_model_1/conv2d_transpose_9/stack:output:0Tsource_separation_model_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0=source_separation_model_1/conv2d_transpose_8/BiasAdd:output:0*
T0*0
_output_shapes
:?????????z?*
paddingVALID*
strides
?
Csource_separation_model_1/conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOpLsource_separation_model_1_conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
4source_separation_model_1/conv2d_transpose_9/BiasAddBiasAddFsource_separation_model_1/conv2d_transpose_9/conv2d_transpose:output:0Ksource_separation_model_1/conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z??
:source_separation_model_1/dense_8/Tensordot/ReadVariableOpReadVariableOpCsource_separation_model_1_dense_8_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0z
0source_separation_model_1/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
0source_separation_model_1/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          ?
1source_separation_model_1/dense_8/Tensordot/ShapeShape4source_separation_model_1/dense_6/Relu:activations:0*
T0*
_output_shapes
:{
9source_separation_model_1/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4source_separation_model_1/dense_8/Tensordot/GatherV2GatherV2:source_separation_model_1/dense_8/Tensordot/Shape:output:09source_separation_model_1/dense_8/Tensordot/free:output:0Bsource_separation_model_1/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
;source_separation_model_1/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6source_separation_model_1/dense_8/Tensordot/GatherV2_1GatherV2:source_separation_model_1/dense_8/Tensordot/Shape:output:09source_separation_model_1/dense_8/Tensordot/axes:output:0Dsource_separation_model_1/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
1source_separation_model_1/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
0source_separation_model_1/dense_8/Tensordot/ProdProd=source_separation_model_1/dense_8/Tensordot/GatherV2:output:0:source_separation_model_1/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3source_separation_model_1/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
2source_separation_model_1/dense_8/Tensordot/Prod_1Prod?source_separation_model_1/dense_8/Tensordot/GatherV2_1:output:0<source_separation_model_1/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7source_separation_model_1/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2source_separation_model_1/dense_8/Tensordot/concatConcatV29source_separation_model_1/dense_8/Tensordot/free:output:09source_separation_model_1/dense_8/Tensordot/axes:output:0@source_separation_model_1/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1source_separation_model_1/dense_8/Tensordot/stackPack9source_separation_model_1/dense_8/Tensordot/Prod:output:0;source_separation_model_1/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
5source_separation_model_1/dense_8/Tensordot/transpose	Transpose4source_separation_model_1/dense_6/Relu:activations:0;source_separation_model_1/dense_8/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????]??
3source_separation_model_1/dense_8/Tensordot/ReshapeReshape9source_separation_model_1/dense_8/Tensordot/transpose:y:0:source_separation_model_1/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
2source_separation_model_1/dense_8/Tensordot/MatMulMatMul<source_separation_model_1/dense_8/Tensordot/Reshape:output:0Bsource_separation_model_1/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????}
3source_separation_model_1/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:{
9source_separation_model_1/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4source_separation_model_1/dense_8/Tensordot/concat_1ConcatV2=source_separation_model_1/dense_8/Tensordot/GatherV2:output:0<source_separation_model_1/dense_8/Tensordot/Const_2:output:0Bsource_separation_model_1/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
+source_separation_model_1/dense_8/TensordotReshape<source_separation_model_1/dense_8/Tensordot/MatMul:product:0=source_separation_model_1/dense_8/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????]?
8source_separation_model_1/dense_8/BiasAdd/ReadVariableOpReadVariableOpAsource_separation_model_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
)source_separation_model_1/dense_8/BiasAddBiasAdd4source_separation_model_1/dense_8/Tensordot:output:0@source_separation_model_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]?
&source_separation_model_1/dense_8/ReluRelu2source_separation_model_1/dense_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????]?
3source_separation_model_1/conv2d_transpose_10/ShapeShape4source_separation_model_1/dense_8/Relu:activations:0*
T0*
_output_shapes
:?
Asource_separation_model_1/conv2d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Csource_separation_model_1/conv2d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Csource_separation_model_1/conv2d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;source_separation_model_1/conv2d_transpose_10/strided_sliceStridedSlice<source_separation_model_1/conv2d_transpose_10/Shape:output:0Jsource_separation_model_1/conv2d_transpose_10/strided_slice/stack:output:0Lsource_separation_model_1/conv2d_transpose_10/strided_slice/stack_1:output:0Lsource_separation_model_1/conv2d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5source_separation_model_1/conv2d_transpose_10/stack/1Const*
_output_shapes
: *
dtype0*
value	B :zw
5source_separation_model_1/conv2d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :w
5source_separation_model_1/conv2d_transpose_10/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
3source_separation_model_1/conv2d_transpose_10/stackPackDsource_separation_model_1/conv2d_transpose_10/strided_slice:output:0>source_separation_model_1/conv2d_transpose_10/stack/1:output:0>source_separation_model_1/conv2d_transpose_10/stack/2:output:0>source_separation_model_1/conv2d_transpose_10/stack/3:output:0*
N*
T0*
_output_shapes
:?
Csource_separation_model_1/conv2d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Esource_separation_model_1/conv2d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Esource_separation_model_1/conv2d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
=source_separation_model_1/conv2d_transpose_10/strided_slice_1StridedSlice<source_separation_model_1/conv2d_transpose_10/stack:output:0Lsource_separation_model_1/conv2d_transpose_10/strided_slice_1/stack:output:0Nsource_separation_model_1/conv2d_transpose_10/strided_slice_1/stack_1:output:0Nsource_separation_model_1/conv2d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Msource_separation_model_1/conv2d_transpose_10/conv2d_transpose/ReadVariableOpReadVariableOpVsource_separation_model_1_conv2d_transpose_10_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
>source_separation_model_1/conv2d_transpose_10/conv2d_transposeConv2DBackpropInput<source_separation_model_1/conv2d_transpose_10/stack:output:0Usource_separation_model_1/conv2d_transpose_10/conv2d_transpose/ReadVariableOp:value:04source_separation_model_1/dense_8/Relu:activations:0*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
?
Dsource_separation_model_1/conv2d_transpose_10/BiasAdd/ReadVariableOpReadVariableOpMsource_separation_model_1_conv2d_transpose_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
5source_separation_model_1/conv2d_transpose_10/BiasAddBiasAddGsource_separation_model_1/conv2d_transpose_10/conv2d_transpose:output:0Lsource_separation_model_1/conv2d_transpose_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z?
3source_separation_model_1/conv2d_transpose_11/ShapeShape>source_separation_model_1/conv2d_transpose_10/BiasAdd:output:0*
T0*
_output_shapes
:?
Asource_separation_model_1/conv2d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Csource_separation_model_1/conv2d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Csource_separation_model_1/conv2d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;source_separation_model_1/conv2d_transpose_11/strided_sliceStridedSlice<source_separation_model_1/conv2d_transpose_11/Shape:output:0Jsource_separation_model_1/conv2d_transpose_11/strided_slice/stack:output:0Lsource_separation_model_1/conv2d_transpose_11/strided_slice/stack_1:output:0Lsource_separation_model_1/conv2d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5source_separation_model_1/conv2d_transpose_11/stack/1Const*
_output_shapes
: *
dtype0*
value	B :zx
5source_separation_model_1/conv2d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?w
5source_separation_model_1/conv2d_transpose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
3source_separation_model_1/conv2d_transpose_11/stackPackDsource_separation_model_1/conv2d_transpose_11/strided_slice:output:0>source_separation_model_1/conv2d_transpose_11/stack/1:output:0>source_separation_model_1/conv2d_transpose_11/stack/2:output:0>source_separation_model_1/conv2d_transpose_11/stack/3:output:0*
N*
T0*
_output_shapes
:?
Csource_separation_model_1/conv2d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Esource_separation_model_1/conv2d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Esource_separation_model_1/conv2d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
=source_separation_model_1/conv2d_transpose_11/strided_slice_1StridedSlice<source_separation_model_1/conv2d_transpose_11/stack:output:0Lsource_separation_model_1/conv2d_transpose_11/strided_slice_1/stack:output:0Nsource_separation_model_1/conv2d_transpose_11/strided_slice_1/stack_1:output:0Nsource_separation_model_1/conv2d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Msource_separation_model_1/conv2d_transpose_11/conv2d_transpose/ReadVariableOpReadVariableOpVsource_separation_model_1_conv2d_transpose_11_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
>source_separation_model_1/conv2d_transpose_11/conv2d_transposeConv2DBackpropInput<source_separation_model_1/conv2d_transpose_11/stack:output:0Usource_separation_model_1/conv2d_transpose_11/conv2d_transpose/ReadVariableOp:value:0>source_separation_model_1/conv2d_transpose_10/BiasAdd:output:0*
T0*0
_output_shapes
:?????????z?*
paddingVALID*
strides
?
Dsource_separation_model_1/conv2d_transpose_11/BiasAdd/ReadVariableOpReadVariableOpMsource_separation_model_1_conv2d_transpose_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
5source_separation_model_1/conv2d_transpose_11/BiasAddBiasAddGsource_separation_model_1/conv2d_transpose_11/conv2d_transpose:output:0Lsource_separation_model_1/conv2d_transpose_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z??
:source_separation_model_1/dense_9/Tensordot/ReadVariableOpReadVariableOpCsource_separation_model_1_dense_9_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0z
0source_separation_model_1/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
0source_separation_model_1/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          ?
1source_separation_model_1/dense_9/Tensordot/ShapeShape4source_separation_model_1/dense_6/Relu:activations:0*
T0*
_output_shapes
:{
9source_separation_model_1/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4source_separation_model_1/dense_9/Tensordot/GatherV2GatherV2:source_separation_model_1/dense_9/Tensordot/Shape:output:09source_separation_model_1/dense_9/Tensordot/free:output:0Bsource_separation_model_1/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
;source_separation_model_1/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6source_separation_model_1/dense_9/Tensordot/GatherV2_1GatherV2:source_separation_model_1/dense_9/Tensordot/Shape:output:09source_separation_model_1/dense_9/Tensordot/axes:output:0Dsource_separation_model_1/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
1source_separation_model_1/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
0source_separation_model_1/dense_9/Tensordot/ProdProd=source_separation_model_1/dense_9/Tensordot/GatherV2:output:0:source_separation_model_1/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3source_separation_model_1/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
2source_separation_model_1/dense_9/Tensordot/Prod_1Prod?source_separation_model_1/dense_9/Tensordot/GatherV2_1:output:0<source_separation_model_1/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7source_separation_model_1/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2source_separation_model_1/dense_9/Tensordot/concatConcatV29source_separation_model_1/dense_9/Tensordot/free:output:09source_separation_model_1/dense_9/Tensordot/axes:output:0@source_separation_model_1/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1source_separation_model_1/dense_9/Tensordot/stackPack9source_separation_model_1/dense_9/Tensordot/Prod:output:0;source_separation_model_1/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
5source_separation_model_1/dense_9/Tensordot/transpose	Transpose4source_separation_model_1/dense_6/Relu:activations:0;source_separation_model_1/dense_9/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????]??
3source_separation_model_1/dense_9/Tensordot/ReshapeReshape9source_separation_model_1/dense_9/Tensordot/transpose:y:0:source_separation_model_1/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
2source_separation_model_1/dense_9/Tensordot/MatMulMatMul<source_separation_model_1/dense_9/Tensordot/Reshape:output:0Bsource_separation_model_1/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????}
3source_separation_model_1/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:{
9source_separation_model_1/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4source_separation_model_1/dense_9/Tensordot/concat_1ConcatV2=source_separation_model_1/dense_9/Tensordot/GatherV2:output:0<source_separation_model_1/dense_9/Tensordot/Const_2:output:0Bsource_separation_model_1/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
+source_separation_model_1/dense_9/TensordotReshape<source_separation_model_1/dense_9/Tensordot/MatMul:product:0=source_separation_model_1/dense_9/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????]?
8source_separation_model_1/dense_9/BiasAdd/ReadVariableOpReadVariableOpAsource_separation_model_1_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
)source_separation_model_1/dense_9/BiasAddBiasAdd4source_separation_model_1/dense_9/Tensordot:output:0@source_separation_model_1/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]?
&source_separation_model_1/dense_9/ReluRelu2source_separation_model_1/dense_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????]?
3source_separation_model_1/conv2d_transpose_12/ShapeShape4source_separation_model_1/dense_9/Relu:activations:0*
T0*
_output_shapes
:?
Asource_separation_model_1/conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Csource_separation_model_1/conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Csource_separation_model_1/conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;source_separation_model_1/conv2d_transpose_12/strided_sliceStridedSlice<source_separation_model_1/conv2d_transpose_12/Shape:output:0Jsource_separation_model_1/conv2d_transpose_12/strided_slice/stack:output:0Lsource_separation_model_1/conv2d_transpose_12/strided_slice/stack_1:output:0Lsource_separation_model_1/conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5source_separation_model_1/conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :zw
5source_separation_model_1/conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :w
5source_separation_model_1/conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
3source_separation_model_1/conv2d_transpose_12/stackPackDsource_separation_model_1/conv2d_transpose_12/strided_slice:output:0>source_separation_model_1/conv2d_transpose_12/stack/1:output:0>source_separation_model_1/conv2d_transpose_12/stack/2:output:0>source_separation_model_1/conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:?
Csource_separation_model_1/conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Esource_separation_model_1/conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Esource_separation_model_1/conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
=source_separation_model_1/conv2d_transpose_12/strided_slice_1StridedSlice<source_separation_model_1/conv2d_transpose_12/stack:output:0Lsource_separation_model_1/conv2d_transpose_12/strided_slice_1/stack:output:0Nsource_separation_model_1/conv2d_transpose_12/strided_slice_1/stack_1:output:0Nsource_separation_model_1/conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Msource_separation_model_1/conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOpVsource_separation_model_1_conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
>source_separation_model_1/conv2d_transpose_12/conv2d_transposeConv2DBackpropInput<source_separation_model_1/conv2d_transpose_12/stack:output:0Usource_separation_model_1/conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:04source_separation_model_1/dense_9/Relu:activations:0*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
?
Dsource_separation_model_1/conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOpMsource_separation_model_1_conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
5source_separation_model_1/conv2d_transpose_12/BiasAddBiasAddGsource_separation_model_1/conv2d_transpose_12/conv2d_transpose:output:0Lsource_separation_model_1/conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z?
3source_separation_model_1/conv2d_transpose_13/ShapeShape>source_separation_model_1/conv2d_transpose_12/BiasAdd:output:0*
T0*
_output_shapes
:?
Asource_separation_model_1/conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Csource_separation_model_1/conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Csource_separation_model_1/conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;source_separation_model_1/conv2d_transpose_13/strided_sliceStridedSlice<source_separation_model_1/conv2d_transpose_13/Shape:output:0Jsource_separation_model_1/conv2d_transpose_13/strided_slice/stack:output:0Lsource_separation_model_1/conv2d_transpose_13/strided_slice/stack_1:output:0Lsource_separation_model_1/conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5source_separation_model_1/conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value	B :zx
5source_separation_model_1/conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?w
5source_separation_model_1/conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
3source_separation_model_1/conv2d_transpose_13/stackPackDsource_separation_model_1/conv2d_transpose_13/strided_slice:output:0>source_separation_model_1/conv2d_transpose_13/stack/1:output:0>source_separation_model_1/conv2d_transpose_13/stack/2:output:0>source_separation_model_1/conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:?
Csource_separation_model_1/conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Esource_separation_model_1/conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Esource_separation_model_1/conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
=source_separation_model_1/conv2d_transpose_13/strided_slice_1StridedSlice<source_separation_model_1/conv2d_transpose_13/stack:output:0Lsource_separation_model_1/conv2d_transpose_13/strided_slice_1/stack:output:0Nsource_separation_model_1/conv2d_transpose_13/strided_slice_1/stack_1:output:0Nsource_separation_model_1/conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Msource_separation_model_1/conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOpVsource_separation_model_1_conv2d_transpose_13_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
>source_separation_model_1/conv2d_transpose_13/conv2d_transposeConv2DBackpropInput<source_separation_model_1/conv2d_transpose_13/stack:output:0Usource_separation_model_1/conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:0>source_separation_model_1/conv2d_transpose_12/BiasAdd:output:0*
T0*0
_output_shapes
:?????????z?*
paddingVALID*
strides
?
Dsource_separation_model_1/conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOpMsource_separation_model_1_conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
5source_separation_model_1/conv2d_transpose_13/BiasAddBiasAddGsource_separation_model_1/conv2d_transpose_13/conv2d_transpose:output:0Lsource_separation_model_1/conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z??
;source_separation_model_1/dense_10/Tensordot/ReadVariableOpReadVariableOpDsource_separation_model_1_dense_10_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0{
1source_separation_model_1/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
1source_separation_model_1/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          ?
2source_separation_model_1/dense_10/Tensordot/ShapeShape4source_separation_model_1/dense_6/Relu:activations:0*
T0*
_output_shapes
:|
:source_separation_model_1/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5source_separation_model_1/dense_10/Tensordot/GatherV2GatherV2;source_separation_model_1/dense_10/Tensordot/Shape:output:0:source_separation_model_1/dense_10/Tensordot/free:output:0Csource_separation_model_1/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
<source_separation_model_1/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
7source_separation_model_1/dense_10/Tensordot/GatherV2_1GatherV2;source_separation_model_1/dense_10/Tensordot/Shape:output:0:source_separation_model_1/dense_10/Tensordot/axes:output:0Esource_separation_model_1/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:|
2source_separation_model_1/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
1source_separation_model_1/dense_10/Tensordot/ProdProd>source_separation_model_1/dense_10/Tensordot/GatherV2:output:0;source_separation_model_1/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: ~
4source_separation_model_1/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
3source_separation_model_1/dense_10/Tensordot/Prod_1Prod@source_separation_model_1/dense_10/Tensordot/GatherV2_1:output:0=source_separation_model_1/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: z
8source_separation_model_1/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
3source_separation_model_1/dense_10/Tensordot/concatConcatV2:source_separation_model_1/dense_10/Tensordot/free:output:0:source_separation_model_1/dense_10/Tensordot/axes:output:0Asource_separation_model_1/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
2source_separation_model_1/dense_10/Tensordot/stackPack:source_separation_model_1/dense_10/Tensordot/Prod:output:0<source_separation_model_1/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
6source_separation_model_1/dense_10/Tensordot/transpose	Transpose4source_separation_model_1/dense_6/Relu:activations:0<source_separation_model_1/dense_10/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????]??
4source_separation_model_1/dense_10/Tensordot/ReshapeReshape:source_separation_model_1/dense_10/Tensordot/transpose:y:0;source_separation_model_1/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
3source_separation_model_1/dense_10/Tensordot/MatMulMatMul=source_separation_model_1/dense_10/Tensordot/Reshape:output:0Csource_separation_model_1/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
4source_separation_model_1/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:|
:source_separation_model_1/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5source_separation_model_1/dense_10/Tensordot/concat_1ConcatV2>source_separation_model_1/dense_10/Tensordot/GatherV2:output:0=source_separation_model_1/dense_10/Tensordot/Const_2:output:0Csource_separation_model_1/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
,source_separation_model_1/dense_10/TensordotReshape=source_separation_model_1/dense_10/Tensordot/MatMul:product:0>source_separation_model_1/dense_10/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????]?
9source_separation_model_1/dense_10/BiasAdd/ReadVariableOpReadVariableOpBsource_separation_model_1_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
*source_separation_model_1/dense_10/BiasAddBiasAdd5source_separation_model_1/dense_10/Tensordot:output:0Asource_separation_model_1/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]?
'source_separation_model_1/dense_10/ReluRelu3source_separation_model_1/dense_10/BiasAdd:output:0*
T0*/
_output_shapes
:?????????]?
3source_separation_model_1/conv2d_transpose_14/ShapeShape5source_separation_model_1/dense_10/Relu:activations:0*
T0*
_output_shapes
:?
Asource_separation_model_1/conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Csource_separation_model_1/conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Csource_separation_model_1/conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;source_separation_model_1/conv2d_transpose_14/strided_sliceStridedSlice<source_separation_model_1/conv2d_transpose_14/Shape:output:0Jsource_separation_model_1/conv2d_transpose_14/strided_slice/stack:output:0Lsource_separation_model_1/conv2d_transpose_14/strided_slice/stack_1:output:0Lsource_separation_model_1/conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5source_separation_model_1/conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :zw
5source_separation_model_1/conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :w
5source_separation_model_1/conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
3source_separation_model_1/conv2d_transpose_14/stackPackDsource_separation_model_1/conv2d_transpose_14/strided_slice:output:0>source_separation_model_1/conv2d_transpose_14/stack/1:output:0>source_separation_model_1/conv2d_transpose_14/stack/2:output:0>source_separation_model_1/conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:?
Csource_separation_model_1/conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Esource_separation_model_1/conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Esource_separation_model_1/conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
=source_separation_model_1/conv2d_transpose_14/strided_slice_1StridedSlice<source_separation_model_1/conv2d_transpose_14/stack:output:0Lsource_separation_model_1/conv2d_transpose_14/strided_slice_1/stack:output:0Nsource_separation_model_1/conv2d_transpose_14/strided_slice_1/stack_1:output:0Nsource_separation_model_1/conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Msource_separation_model_1/conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOpVsource_separation_model_1_conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
>source_separation_model_1/conv2d_transpose_14/conv2d_transposeConv2DBackpropInput<source_separation_model_1/conv2d_transpose_14/stack:output:0Usource_separation_model_1/conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:05source_separation_model_1/dense_10/Relu:activations:0*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
?
Dsource_separation_model_1/conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOpMsource_separation_model_1_conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
5source_separation_model_1/conv2d_transpose_14/BiasAddBiasAddGsource_separation_model_1/conv2d_transpose_14/conv2d_transpose:output:0Lsource_separation_model_1/conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z?
3source_separation_model_1/conv2d_transpose_15/ShapeShape>source_separation_model_1/conv2d_transpose_14/BiasAdd:output:0*
T0*
_output_shapes
:?
Asource_separation_model_1/conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Csource_separation_model_1/conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Csource_separation_model_1/conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;source_separation_model_1/conv2d_transpose_15/strided_sliceStridedSlice<source_separation_model_1/conv2d_transpose_15/Shape:output:0Jsource_separation_model_1/conv2d_transpose_15/strided_slice/stack:output:0Lsource_separation_model_1/conv2d_transpose_15/strided_slice/stack_1:output:0Lsource_separation_model_1/conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5source_separation_model_1/conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :zx
5source_separation_model_1/conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?w
5source_separation_model_1/conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
3source_separation_model_1/conv2d_transpose_15/stackPackDsource_separation_model_1/conv2d_transpose_15/strided_slice:output:0>source_separation_model_1/conv2d_transpose_15/stack/1:output:0>source_separation_model_1/conv2d_transpose_15/stack/2:output:0>source_separation_model_1/conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:?
Csource_separation_model_1/conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Esource_separation_model_1/conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Esource_separation_model_1/conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
=source_separation_model_1/conv2d_transpose_15/strided_slice_1StridedSlice<source_separation_model_1/conv2d_transpose_15/stack:output:0Lsource_separation_model_1/conv2d_transpose_15/strided_slice_1/stack:output:0Nsource_separation_model_1/conv2d_transpose_15/strided_slice_1/stack_1:output:0Nsource_separation_model_1/conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Msource_separation_model_1/conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOpVsource_separation_model_1_conv2d_transpose_15_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
>source_separation_model_1/conv2d_transpose_15/conv2d_transposeConv2DBackpropInput<source_separation_model_1/conv2d_transpose_15/stack:output:0Usource_separation_model_1/conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0>source_separation_model_1/conv2d_transpose_14/BiasAdd:output:0*
T0*0
_output_shapes
:?????????z?*
paddingVALID*
strides
?
Dsource_separation_model_1/conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOpMsource_separation_model_1_conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
5source_separation_model_1/conv2d_transpose_15/BiasAddBiasAddGsource_separation_model_1/conv2d_transpose_15/conv2d_transpose:output:0Lsource_separation_model_1/conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z?u
3source_separation_model_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
.source_separation_model_1/concatenate_1/concatConcatV2=source_separation_model_1/conv2d_transpose_9/BiasAdd:output:0>source_separation_model_1/conv2d_transpose_11/BiasAdd:output:0>source_separation_model_1/conv2d_transpose_13/BiasAdd:output:0>source_separation_model_1/conv2d_transpose_15/BiasAdd:output:0<source_separation_model_1/concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????z?x?
;source_separation_model_1/dense_11/Tensordot/ReadVariableOpReadVariableOpDsource_separation_model_1_dense_11_tensordot_readvariableop_resource*
_output_shapes

:x*
dtype0{
1source_separation_model_1/dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
1source_separation_model_1/dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          ?
2source_separation_model_1/dense_11/Tensordot/ShapeShape7source_separation_model_1/concatenate_1/concat:output:0*
T0*
_output_shapes
:|
:source_separation_model_1/dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5source_separation_model_1/dense_11/Tensordot/GatherV2GatherV2;source_separation_model_1/dense_11/Tensordot/Shape:output:0:source_separation_model_1/dense_11/Tensordot/free:output:0Csource_separation_model_1/dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
<source_separation_model_1/dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
7source_separation_model_1/dense_11/Tensordot/GatherV2_1GatherV2;source_separation_model_1/dense_11/Tensordot/Shape:output:0:source_separation_model_1/dense_11/Tensordot/axes:output:0Esource_separation_model_1/dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:|
2source_separation_model_1/dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
1source_separation_model_1/dense_11/Tensordot/ProdProd>source_separation_model_1/dense_11/Tensordot/GatherV2:output:0;source_separation_model_1/dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: ~
4source_separation_model_1/dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
3source_separation_model_1/dense_11/Tensordot/Prod_1Prod@source_separation_model_1/dense_11/Tensordot/GatherV2_1:output:0=source_separation_model_1/dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: z
8source_separation_model_1/dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
3source_separation_model_1/dense_11/Tensordot/concatConcatV2:source_separation_model_1/dense_11/Tensordot/free:output:0:source_separation_model_1/dense_11/Tensordot/axes:output:0Asource_separation_model_1/dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
2source_separation_model_1/dense_11/Tensordot/stackPack:source_separation_model_1/dense_11/Tensordot/Prod:output:0<source_separation_model_1/dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
6source_separation_model_1/dense_11/Tensordot/transpose	Transpose7source_separation_model_1/concatenate_1/concat:output:0<source_separation_model_1/dense_11/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????z?x?
4source_separation_model_1/dense_11/Tensordot/ReshapeReshape:source_separation_model_1/dense_11/Tensordot/transpose:y:0;source_separation_model_1/dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
3source_separation_model_1/dense_11/Tensordot/MatMulMatMul=source_separation_model_1/dense_11/Tensordot/Reshape:output:0Csource_separation_model_1/dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
4source_separation_model_1/dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:|
:source_separation_model_1/dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5source_separation_model_1/dense_11/Tensordot/concat_1ConcatV2>source_separation_model_1/dense_11/Tensordot/GatherV2:output:0=source_separation_model_1/dense_11/Tensordot/Const_2:output:0Csource_separation_model_1/dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
,source_separation_model_1/dense_11/TensordotReshape=source_separation_model_1/dense_11/Tensordot/MatMul:product:0>source_separation_model_1/dense_11/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????z??
9source_separation_model_1/dense_11/BiasAdd/ReadVariableOpReadVariableOpBsource_separation_model_1_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
*source_separation_model_1/dense_11/BiasAddBiasAdd5source_separation_model_1/dense_11/Tensordot:output:0Asource_separation_model_1/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z??
'source_separation_model_1/dense_11/ReluRelu3source_separation_model_1/dense_11/BiasAdd:output:0*
T0*0
_output_shapes
:?????????z??
IdentityIdentity5source_separation_model_1/dense_11/Relu:activations:0^NoOp*
T0*0
_output_shapes
:?????????z??
NoOpNoOp:^source_separation_model_1/conv2d_2/BiasAdd/ReadVariableOp9^source_separation_model_1/conv2d_2/Conv2D/ReadVariableOp:^source_separation_model_1/conv2d_3/BiasAdd/ReadVariableOp9^source_separation_model_1/conv2d_3/Conv2D/ReadVariableOpE^source_separation_model_1/conv2d_transpose_10/BiasAdd/ReadVariableOpN^source_separation_model_1/conv2d_transpose_10/conv2d_transpose/ReadVariableOpE^source_separation_model_1/conv2d_transpose_11/BiasAdd/ReadVariableOpN^source_separation_model_1/conv2d_transpose_11/conv2d_transpose/ReadVariableOpE^source_separation_model_1/conv2d_transpose_12/BiasAdd/ReadVariableOpN^source_separation_model_1/conv2d_transpose_12/conv2d_transpose/ReadVariableOpE^source_separation_model_1/conv2d_transpose_13/BiasAdd/ReadVariableOpN^source_separation_model_1/conv2d_transpose_13/conv2d_transpose/ReadVariableOpE^source_separation_model_1/conv2d_transpose_14/BiasAdd/ReadVariableOpN^source_separation_model_1/conv2d_transpose_14/conv2d_transpose/ReadVariableOpE^source_separation_model_1/conv2d_transpose_15/BiasAdd/ReadVariableOpN^source_separation_model_1/conv2d_transpose_15/conv2d_transpose/ReadVariableOpD^source_separation_model_1/conv2d_transpose_8/BiasAdd/ReadVariableOpM^source_separation_model_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOpD^source_separation_model_1/conv2d_transpose_9/BiasAdd/ReadVariableOpM^source_separation_model_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:^source_separation_model_1/dense_10/BiasAdd/ReadVariableOp<^source_separation_model_1/dense_10/Tensordot/ReadVariableOp:^source_separation_model_1/dense_11/BiasAdd/ReadVariableOp<^source_separation_model_1/dense_11/Tensordot/ReadVariableOp9^source_separation_model_1/dense_6/BiasAdd/ReadVariableOp;^source_separation_model_1/dense_6/Tensordot/ReadVariableOp9^source_separation_model_1/dense_7/BiasAdd/ReadVariableOp;^source_separation_model_1/dense_7/Tensordot/ReadVariableOp9^source_separation_model_1/dense_8/BiasAdd/ReadVariableOp;^source_separation_model_1/dense_8/Tensordot/ReadVariableOp9^source_separation_model_1/dense_9/BiasAdd/ReadVariableOp;^source_separation_model_1/dense_9/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:?????????z?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2v
9source_separation_model_1/conv2d_2/BiasAdd/ReadVariableOp9source_separation_model_1/conv2d_2/BiasAdd/ReadVariableOp2t
8source_separation_model_1/conv2d_2/Conv2D/ReadVariableOp8source_separation_model_1/conv2d_2/Conv2D/ReadVariableOp2v
9source_separation_model_1/conv2d_3/BiasAdd/ReadVariableOp9source_separation_model_1/conv2d_3/BiasAdd/ReadVariableOp2t
8source_separation_model_1/conv2d_3/Conv2D/ReadVariableOp8source_separation_model_1/conv2d_3/Conv2D/ReadVariableOp2?
Dsource_separation_model_1/conv2d_transpose_10/BiasAdd/ReadVariableOpDsource_separation_model_1/conv2d_transpose_10/BiasAdd/ReadVariableOp2?
Msource_separation_model_1/conv2d_transpose_10/conv2d_transpose/ReadVariableOpMsource_separation_model_1/conv2d_transpose_10/conv2d_transpose/ReadVariableOp2?
Dsource_separation_model_1/conv2d_transpose_11/BiasAdd/ReadVariableOpDsource_separation_model_1/conv2d_transpose_11/BiasAdd/ReadVariableOp2?
Msource_separation_model_1/conv2d_transpose_11/conv2d_transpose/ReadVariableOpMsource_separation_model_1/conv2d_transpose_11/conv2d_transpose/ReadVariableOp2?
Dsource_separation_model_1/conv2d_transpose_12/BiasAdd/ReadVariableOpDsource_separation_model_1/conv2d_transpose_12/BiasAdd/ReadVariableOp2?
Msource_separation_model_1/conv2d_transpose_12/conv2d_transpose/ReadVariableOpMsource_separation_model_1/conv2d_transpose_12/conv2d_transpose/ReadVariableOp2?
Dsource_separation_model_1/conv2d_transpose_13/BiasAdd/ReadVariableOpDsource_separation_model_1/conv2d_transpose_13/BiasAdd/ReadVariableOp2?
Msource_separation_model_1/conv2d_transpose_13/conv2d_transpose/ReadVariableOpMsource_separation_model_1/conv2d_transpose_13/conv2d_transpose/ReadVariableOp2?
Dsource_separation_model_1/conv2d_transpose_14/BiasAdd/ReadVariableOpDsource_separation_model_1/conv2d_transpose_14/BiasAdd/ReadVariableOp2?
Msource_separation_model_1/conv2d_transpose_14/conv2d_transpose/ReadVariableOpMsource_separation_model_1/conv2d_transpose_14/conv2d_transpose/ReadVariableOp2?
Dsource_separation_model_1/conv2d_transpose_15/BiasAdd/ReadVariableOpDsource_separation_model_1/conv2d_transpose_15/BiasAdd/ReadVariableOp2?
Msource_separation_model_1/conv2d_transpose_15/conv2d_transpose/ReadVariableOpMsource_separation_model_1/conv2d_transpose_15/conv2d_transpose/ReadVariableOp2?
Csource_separation_model_1/conv2d_transpose_8/BiasAdd/ReadVariableOpCsource_separation_model_1/conv2d_transpose_8/BiasAdd/ReadVariableOp2?
Lsource_separation_model_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOpLsource_separation_model_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOp2?
Csource_separation_model_1/conv2d_transpose_9/BiasAdd/ReadVariableOpCsource_separation_model_1/conv2d_transpose_9/BiasAdd/ReadVariableOp2?
Lsource_separation_model_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOpLsource_separation_model_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOp2v
9source_separation_model_1/dense_10/BiasAdd/ReadVariableOp9source_separation_model_1/dense_10/BiasAdd/ReadVariableOp2z
;source_separation_model_1/dense_10/Tensordot/ReadVariableOp;source_separation_model_1/dense_10/Tensordot/ReadVariableOp2v
9source_separation_model_1/dense_11/BiasAdd/ReadVariableOp9source_separation_model_1/dense_11/BiasAdd/ReadVariableOp2z
;source_separation_model_1/dense_11/Tensordot/ReadVariableOp;source_separation_model_1/dense_11/Tensordot/ReadVariableOp2t
8source_separation_model_1/dense_6/BiasAdd/ReadVariableOp8source_separation_model_1/dense_6/BiasAdd/ReadVariableOp2x
:source_separation_model_1/dense_6/Tensordot/ReadVariableOp:source_separation_model_1/dense_6/Tensordot/ReadVariableOp2t
8source_separation_model_1/dense_7/BiasAdd/ReadVariableOp8source_separation_model_1/dense_7/BiasAdd/ReadVariableOp2x
:source_separation_model_1/dense_7/Tensordot/ReadVariableOp:source_separation_model_1/dense_7/Tensordot/ReadVariableOp2t
8source_separation_model_1/dense_8/BiasAdd/ReadVariableOp8source_separation_model_1/dense_8/BiasAdd/ReadVariableOp2x
:source_separation_model_1/dense_8/Tensordot/ReadVariableOp:source_separation_model_1/dense_8/Tensordot/ReadVariableOp2t
8source_separation_model_1/dense_9/BiasAdd/ReadVariableOp8source_separation_model_1/dense_9/BiasAdd/ReadVariableOp2x
:source_separation_model_1/dense_9/Tensordot/ReadVariableOp:source_separation_model_1/dense_9/Tensordot/ReadVariableOp:Y U
0
_output_shapes
:?????????z?
!
_user_specified_name	input_1
?
?
Q__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_21061951

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
6__inference_conv2d_transpose_15_layer_call_fn_21062281

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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_21060570x
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
?
?
E__inference_dense_9_layer_call_and_return_conditional_losses_21061677

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
Q__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_21062240

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
?
?
P__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_21060291

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
?^
?
W__inference_source_separation_model_1_layer_call_and_return_conditional_losses_21061007
input_1,
conv2d_2_21060925:?
conv2d_2_21060927:+
conv2d_3_21060930:
conv2d_3_21060932:#
dense_6_21060935:	?
dense_6_21060937:	?#
dense_7_21060940:	?
dense_7_21060942:5
conv2d_transpose_8_21060945:)
conv2d_transpose_8_21060947:6
conv2d_transpose_9_21060950:?)
conv2d_transpose_9_21060952:#
dense_8_21060955:	?
dense_8_21060957:6
conv2d_transpose_10_21060960:*
conv2d_transpose_10_21060962:7
conv2d_transpose_11_21060965:?*
conv2d_transpose_11_21060967:#
dense_9_21060970:	?
dense_9_21060972:6
conv2d_transpose_12_21060975:*
conv2d_transpose_12_21060977:7
conv2d_transpose_13_21060980:?*
conv2d_transpose_13_21060982:$
dense_10_21060985:	?
dense_10_21060987:6
conv2d_transpose_14_21060990:*
conv2d_transpose_14_21060992:7
conv2d_transpose_15_21060995:?*
conv2d_transpose_15_21060997:#
dense_11_21061001:x
dense_11_21061003:
identity?? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?+conv2d_transpose_10/StatefulPartitionedCall?+conv2d_transpose_11/StatefulPartitionedCall?+conv2d_transpose_12/StatefulPartitionedCall?+conv2d_transpose_13/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?+conv2d_transpose_15/StatefulPartitionedCall?*conv2d_transpose_8/StatefulPartitionedCall?*conv2d_transpose_9/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_2_21060925conv2d_2_21060927*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_21060145?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_21060930conv2d_3_21060932*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_21060161?
dense_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0dense_6_21060935dense_6_21060937*
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
GPU 2J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_21060198?
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_21060940dense_7_21060942*
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
E__inference_dense_7_layer_call_and_return_conditional_losses_21060235?
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0conv2d_transpose_8_21060945conv2d_transpose_8_21060947*
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
GPU 2J 8? *Y
fTRR
P__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_21060263?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0conv2d_transpose_9_21060950conv2d_transpose_9_21060952*
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
GPU 2J 8? *Y
fTRR
P__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_21060291?
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_8_21060955dense_8_21060957*
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
E__inference_dense_8_layer_call_and_return_conditional_losses_21060328?
+conv2d_transpose_10/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0conv2d_transpose_10_21060960conv2d_transpose_10_21060962*
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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_21060356?
+conv2d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_10/StatefulPartitionedCall:output:0conv2d_transpose_11_21060965conv2d_transpose_11_21060967*
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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_21060384?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_9_21060970dense_9_21060972*
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
E__inference_dense_9_layer_call_and_return_conditional_losses_21060421?
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0conv2d_transpose_12_21060975conv2d_transpose_12_21060977*
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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_21060449?
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0conv2d_transpose_13_21060980conv2d_transpose_13_21060982*
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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_21060477?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_10_21060985dense_10_21060987*
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
GPU 2J 8? *O
fJRH
F__inference_dense_10_layer_call_and_return_conditional_losses_21060514?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0conv2d_transpose_14_21060990conv2d_transpose_14_21060992*
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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_21060542?
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0conv2d_transpose_15_21060995conv2d_transpose_15_21060997*
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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_21060570?
concatenate_1/PartitionedCallPartitionedCall3conv2d_transpose_9/StatefulPartitionedCall:output:04conv2d_transpose_11/StatefulPartitionedCall:output:04conv2d_transpose_13/StatefulPartitionedCall:output:04conv2d_transpose_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?x* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concatenate_1_layer_call_and_return_conditional_losses_21060585?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_11_21061001dense_11_21061003*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_21060618?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????z??
NoOpNoOp!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall,^conv2d_transpose_10/StatefulPartitionedCall,^conv2d_transpose_11/StatefulPartitionedCall,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall+^conv2d_transpose_8/StatefulPartitionedCall+^conv2d_transpose_9/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:?????????z?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2Z
+conv2d_transpose_10/StatefulPartitionedCall+conv2d_transpose_10/StatefulPartitionedCall2Z
+conv2d_transpose_11/StatefulPartitionedCall+conv2d_transpose_11/StatefulPartitionedCall2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:Y U
0
_output_shapes
:?????????z?
!
_user_specified_name	input_1
?
?
Q__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_21060542

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
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_21062318

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
*__inference_dense_6_layer_call_fn_21061526

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
GPU 2J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_21060198x
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
?#
?
Q__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_21059977

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
+__inference_conv2d_3_layer_call_fn_21061507

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
GPU 2J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_21060161w
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
?
?
6__inference_conv2d_transpose_12_layer_call_fn_21062047

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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_21060449w
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
+__inference_conv2d_2_layer_call_fn_21061488

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
GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_21060145w
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
?
?
E__inference_dense_6_layer_call_and_return_conditional_losses_21060198

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
?
?
6__inference_conv2d_transpose_13_layer_call_fn_21062125

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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_21060477x
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
?
?
6__inference_conv2d_transpose_14_layer_call_fn_21062203

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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_21060542w
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
?
?
Q__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_21060384

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
Q__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_21062263

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
6__inference_conv2d_transpose_10_layer_call_fn_21061891

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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_21060356w
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
?
?
5__inference_conv2d_transpose_9_layer_call_fn_21061804

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
GPU 2J 8? *Y
fTRR
P__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_21059833?
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
?
?
+__inference_dense_11_layer_call_fn_21062367

inputs
unknown:x
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_21060618x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????z?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????z?x: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????z?x
 
_user_specified_nameinputs
?
?
E__inference_dense_7_layer_call_and_return_conditional_losses_21060235

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
F__inference_conv2d_3_layer_call_and_return_conditional_losses_21061517

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
?
?
&__inference_signature_wrapper_21061078
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

unknown_17:	?

unknown_18:$

unknown_19:

unknown_20:%

unknown_21:?

unknown_22:

unknown_23:	?

unknown_24:$

unknown_25:

unknown_26:%

unknown_27:?

unknown_28:

unknown_29:x

unknown_30:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_21059744x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????z?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:?????????z?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:?????????z?
!
_user_specified_name	input_1
?#
?
Q__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_21060025

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
?
?
F__inference_dense_11_layer_call_and_return_conditional_losses_21062398

inputs3
!tensordot_readvariableop_resource:x-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:x*
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
:?????????z?x?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
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
:?????????z?r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????z?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????z?z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????z?x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:X T
0
_output_shapes
:?????????z?x
 
_user_specified_nameinputs
?
?
*__inference_dense_7_layer_call_fn_21061566

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
GPU 2J 8? *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_21060235w
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
?
?
6__inference_conv2d_transpose_14_layer_call_fn_21062194

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_21060073?
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
Q__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_21062006

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
?
?
P__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_21061873

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
?
?
<__inference_source_separation_model_1_layer_call_fn_21061147
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

unknown_17:	?

unknown_18:$

unknown_19:

unknown_20:%

unknown_21:?

unknown_22:

unknown_23:	?

unknown_24:$

unknown_25:

unknown_26:%

unknown_27:?

unknown_28:

unknown_29:x

unknown_30:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8? *`
f[RY
W__inference_source_separation_model_1_layer_call_and_return_conditional_losses_21060625x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????z?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:?????????z?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:?????????z?

_user_specified_namex
?
?
Q__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_21062185

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
?#
?
Q__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_21062162

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
?O
?
!__inference__traced_save_21062517
file_prefixH
Dsavev2_source_separation_model_1_conv2d_2_kernel_read_readvariableopF
Bsavev2_source_separation_model_1_conv2d_2_bias_read_readvariableopH
Dsavev2_source_separation_model_1_conv2d_3_kernel_read_readvariableopF
Bsavev2_source_separation_model_1_conv2d_3_bias_read_readvariableopG
Csavev2_source_separation_model_1_dense_6_kernel_read_readvariableopE
Asavev2_source_separation_model_1_dense_6_bias_read_readvariableopG
Csavev2_source_separation_model_1_dense_7_kernel_read_readvariableopE
Asavev2_source_separation_model_1_dense_7_bias_read_readvariableopG
Csavev2_source_separation_model_1_dense_8_kernel_read_readvariableopE
Asavev2_source_separation_model_1_dense_8_bias_read_readvariableopG
Csavev2_source_separation_model_1_dense_9_kernel_read_readvariableopE
Asavev2_source_separation_model_1_dense_9_bias_read_readvariableopH
Dsavev2_source_separation_model_1_dense_10_kernel_read_readvariableopF
Bsavev2_source_separation_model_1_dense_10_bias_read_readvariableopR
Nsavev2_source_separation_model_1_conv2d_transpose_8_kernel_read_readvariableopP
Lsavev2_source_separation_model_1_conv2d_transpose_8_bias_read_readvariableopR
Nsavev2_source_separation_model_1_conv2d_transpose_9_kernel_read_readvariableopP
Lsavev2_source_separation_model_1_conv2d_transpose_9_bias_read_readvariableopS
Osavev2_source_separation_model_1_conv2d_transpose_10_kernel_read_readvariableopQ
Msavev2_source_separation_model_1_conv2d_transpose_10_bias_read_readvariableopS
Osavev2_source_separation_model_1_conv2d_transpose_11_kernel_read_readvariableopQ
Msavev2_source_separation_model_1_conv2d_transpose_11_bias_read_readvariableopS
Osavev2_source_separation_model_1_conv2d_transpose_12_kernel_read_readvariableopQ
Msavev2_source_separation_model_1_conv2d_transpose_12_bias_read_readvariableopS
Osavev2_source_separation_model_1_conv2d_transpose_13_kernel_read_readvariableopQ
Msavev2_source_separation_model_1_conv2d_transpose_13_bias_read_readvariableopS
Osavev2_source_separation_model_1_conv2d_transpose_14_kernel_read_readvariableopQ
Msavev2_source_separation_model_1_conv2d_transpose_14_bias_read_readvariableopS
Osavev2_source_separation_model_1_conv2d_transpose_15_kernel_read_readvariableopQ
Msavev2_source_separation_model_1_conv2d_transpose_15_bias_read_readvariableopH
Dsavev2_source_separation_model_1_dense_11_kernel_read_readvariableopF
Bsavev2_source_separation_model_1_dense_11_bias_read_readvariableop
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
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB)dClass0/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dClass0/bias/.ATTRIBUTES/VARIABLE_VALUEB)dClass1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dClass1/bias/.ATTRIBUTES/VARIABLE_VALUEB)dClass2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dClass2/bias/.ATTRIBUTES/VARIABLE_VALUEB)dClass3/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dClass3/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv3Class0/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv3Class0/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv4Class0/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv4Class0/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv3Class1/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv3Class1/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv4Class1/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv4Class1/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv3Class2/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv3Class2/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv4Class2/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv4Class2/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv3Class3/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv3Class3/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv4Class3/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv4Class3/bias/.ATTRIBUTES/VARIABLE_VALUEB%out/kernel/.ATTRIBUTES/VARIABLE_VALUEB#out/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Dsavev2_source_separation_model_1_conv2d_2_kernel_read_readvariableopBsavev2_source_separation_model_1_conv2d_2_bias_read_readvariableopDsavev2_source_separation_model_1_conv2d_3_kernel_read_readvariableopBsavev2_source_separation_model_1_conv2d_3_bias_read_readvariableopCsavev2_source_separation_model_1_dense_6_kernel_read_readvariableopAsavev2_source_separation_model_1_dense_6_bias_read_readvariableopCsavev2_source_separation_model_1_dense_7_kernel_read_readvariableopAsavev2_source_separation_model_1_dense_7_bias_read_readvariableopCsavev2_source_separation_model_1_dense_8_kernel_read_readvariableopAsavev2_source_separation_model_1_dense_8_bias_read_readvariableopCsavev2_source_separation_model_1_dense_9_kernel_read_readvariableopAsavev2_source_separation_model_1_dense_9_bias_read_readvariableopDsavev2_source_separation_model_1_dense_10_kernel_read_readvariableopBsavev2_source_separation_model_1_dense_10_bias_read_readvariableopNsavev2_source_separation_model_1_conv2d_transpose_8_kernel_read_readvariableopLsavev2_source_separation_model_1_conv2d_transpose_8_bias_read_readvariableopNsavev2_source_separation_model_1_conv2d_transpose_9_kernel_read_readvariableopLsavev2_source_separation_model_1_conv2d_transpose_9_bias_read_readvariableopOsavev2_source_separation_model_1_conv2d_transpose_10_kernel_read_readvariableopMsavev2_source_separation_model_1_conv2d_transpose_10_bias_read_readvariableopOsavev2_source_separation_model_1_conv2d_transpose_11_kernel_read_readvariableopMsavev2_source_separation_model_1_conv2d_transpose_11_bias_read_readvariableopOsavev2_source_separation_model_1_conv2d_transpose_12_kernel_read_readvariableopMsavev2_source_separation_model_1_conv2d_transpose_12_bias_read_readvariableopOsavev2_source_separation_model_1_conv2d_transpose_13_kernel_read_readvariableopMsavev2_source_separation_model_1_conv2d_transpose_13_bias_read_readvariableopOsavev2_source_separation_model_1_conv2d_transpose_14_kernel_read_readvariableopMsavev2_source_separation_model_1_conv2d_transpose_14_bias_read_readvariableopOsavev2_source_separation_model_1_conv2d_transpose_15_kernel_read_readvariableopMsavev2_source_separation_model_1_conv2d_transpose_15_bias_read_readvariableopDsavev2_source_separation_model_1_dense_11_kernel_read_readvariableopBsavev2_source_separation_model_1_dense_11_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 */
dtypes%
#2!?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :?::::	?:?:	?::	?::	?::	?::::?::::?::::?::::?::x:: 2(
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
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::,(
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
::,(
&
_output_shapes
:: 

_output_shapes
::-)
'
_output_shapes
:?: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::-)
'
_output_shapes
:?: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::-)
'
_output_shapes
:?: 

_output_shapes
::$ 

_output_shapes

:x:  

_output_shapes
::!

_output_shapes
: 
?
?
F__inference_dense_10_layer_call_and_return_conditional_losses_21060514

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
?
?
Q__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_21060356

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
P__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_21059833

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
?
?
E__inference_dense_8_layer_call_and_return_conditional_losses_21060328

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

?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_21061498

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
?
?
6__inference_conv2d_transpose_10_layer_call_fn_21061882

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_21059881?
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
?
?
5__inference_conv2d_transpose_8_layer_call_fn_21061735

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
GPU 2J 8? *Y
fTRR
P__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_21060263w
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
?
?
5__inference_conv2d_transpose_8_layer_call_fn_21061726

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
GPU 2J 8? *Y
fTRR
P__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_21059785?
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
Q__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_21059881

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
??
?
W__inference_source_separation_model_1_layer_call_and_return_conditional_losses_21061479
xB
'conv2d_2_conv2d_readvariableop_resource:?6
(conv2d_2_biasadd_readvariableop_resource:A
'conv2d_3_conv2d_readvariableop_resource:6
(conv2d_3_biasadd_readvariableop_resource:<
)dense_6_tensordot_readvariableop_resource:	?6
'dense_6_biasadd_readvariableop_resource:	?<
)dense_7_tensordot_readvariableop_resource:	?5
'dense_7_biasadd_readvariableop_resource:U
;conv2d_transpose_8_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_8_biasadd_readvariableop_resource:V
;conv2d_transpose_9_conv2d_transpose_readvariableop_resource:?@
2conv2d_transpose_9_biasadd_readvariableop_resource:<
)dense_8_tensordot_readvariableop_resource:	?5
'dense_8_biasadd_readvariableop_resource:V
<conv2d_transpose_10_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_10_biasadd_readvariableop_resource:W
<conv2d_transpose_11_conv2d_transpose_readvariableop_resource:?A
3conv2d_transpose_11_biasadd_readvariableop_resource:<
)dense_9_tensordot_readvariableop_resource:	?5
'dense_9_biasadd_readvariableop_resource:V
<conv2d_transpose_12_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_12_biasadd_readvariableop_resource:W
<conv2d_transpose_13_conv2d_transpose_readvariableop_resource:?A
3conv2d_transpose_13_biasadd_readvariableop_resource:=
*dense_10_tensordot_readvariableop_resource:	?6
(dense_10_biasadd_readvariableop_resource:V
<conv2d_transpose_14_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_14_biasadd_readvariableop_resource:W
<conv2d_transpose_15_conv2d_transpose_readvariableop_resource:?A
3conv2d_transpose_15_biasadd_readvariableop_resource:<
*dense_11_tensordot_readvariableop_resource:x6
(dense_11_biasadd_readvariableop_resource:
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?*conv2d_transpose_10/BiasAdd/ReadVariableOp?3conv2d_transpose_10/conv2d_transpose/ReadVariableOp?*conv2d_transpose_11/BiasAdd/ReadVariableOp?3conv2d_transpose_11/conv2d_transpose/ReadVariableOp?*conv2d_transpose_12/BiasAdd/ReadVariableOp?3conv2d_transpose_12/conv2d_transpose/ReadVariableOp?*conv2d_transpose_13/BiasAdd/ReadVariableOp?3conv2d_transpose_13/conv2d_transpose/ReadVariableOp?*conv2d_transpose_14/BiasAdd/ReadVariableOp?3conv2d_transpose_14/conv2d_transpose/ReadVariableOp?*conv2d_transpose_15/BiasAdd/ReadVariableOp?3conv2d_transpose_15/conv2d_transpose/ReadVariableOp?)conv2d_transpose_8/BiasAdd/ReadVariableOp?2conv2d_transpose_8/conv2d_transpose/ReadVariableOp?)conv2d_transpose_9/BiasAdd/ReadVariableOp?2conv2d_transpose_9/conv2d_transpose/ReadVariableOp?dense_10/BiasAdd/ReadVariableOp?!dense_10/Tensordot/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?!dense_11/Tensordot/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp? dense_6/Tensordot/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp? dense_7/Tensordot/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp? dense_8/Tensordot/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp? dense_9/Tensordot/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d_2/Conv2DConv2Dx&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_3/Conv2DConv2Dconv2d_2/BiasAdd:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]*
paddingVALID*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]?
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0`
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          `
dense_6/Tensordot/ShapeShapeconv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:a
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_6/Tensordot/GatherV2_1GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/axes:output:0*dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_6/Tensordot/transpose	Transposeconv2d_3/BiasAdd:output:0!dense_6/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????]?
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?a
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????]??
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????]?i
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????]??
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0`
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          a
dense_7/Tensordot/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:a
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_7/Tensordot/GatherV2_1GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/axes:output:0*dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_7/Tensordot/transpose	Transposedense_6/Relu:activations:0!dense_7/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????]??
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????]?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]h
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????]b
conv2d_transpose_8/ShapeShapedense_7/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_8/strided_sliceStridedSlice!conv2d_transpose_8/Shape:output:0/conv2d_transpose_8/strided_slice/stack:output:01conv2d_transpose_8/strided_slice/stack_1:output:01conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :z\
conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_8/stackPack)conv2d_transpose_8/strided_slice:output:0#conv2d_transpose_8/stack/1:output:0#conv2d_transpose_8/stack/2:output:0#conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_8/strided_slice_1StridedSlice!conv2d_transpose_8/stack:output:01conv2d_transpose_8/strided_slice_1/stack:output:03conv2d_transpose_8/strided_slice_1/stack_1:output:03conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
#conv2d_transpose_8/conv2d_transposeConv2DBackpropInput!conv2d_transpose_8/stack:output:0:conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:0dense_7/Relu:activations:0*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
?
)conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_8/BiasAddBiasAdd,conv2d_transpose_8/conv2d_transpose:output:01conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????zk
conv2d_transpose_9/ShapeShape#conv2d_transpose_8/BiasAdd:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_9/strided_sliceStridedSlice!conv2d_transpose_9/Shape:output:0/conv2d_transpose_9/strided_slice/stack:output:01conv2d_transpose_9/strided_slice/stack_1:output:01conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :z]
conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?\
conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_9/stackPack)conv2d_transpose_9/strided_slice:output:0#conv2d_transpose_9/stack/1:output:0#conv2d_transpose_9/stack/2:output:0#conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_9/strided_slice_1StridedSlice!conv2d_transpose_9/stack:output:01conv2d_transpose_9/strided_slice_1/stack:output:03conv2d_transpose_9/strided_slice_1/stack_1:output:03conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_9_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
#conv2d_transpose_9/conv2d_transposeConv2DBackpropInput!conv2d_transpose_9/stack:output:0:conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose_8/BiasAdd:output:0*
T0*0
_output_shapes
:?????????z?*
paddingVALID*
strides
?
)conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_9/BiasAddBiasAdd,conv2d_transpose_9/conv2d_transpose:output:01conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z??
 dense_8/Tensordot/ReadVariableOpReadVariableOp)dense_8_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0`
dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          a
dense_8/Tensordot/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:a
dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_8/Tensordot/GatherV2GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/free:output:0(dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_8/Tensordot/GatherV2_1GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/axes:output:0*dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_8/Tensordot/ProdProd#dense_8/Tensordot/GatherV2:output:0 dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_8/Tensordot/Prod_1Prod%dense_8/Tensordot/GatherV2_1:output:0"dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_8/Tensordot/concatConcatV2dense_8/Tensordot/free:output:0dense_8/Tensordot/axes:output:0&dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_8/Tensordot/stackPackdense_8/Tensordot/Prod:output:0!dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_8/Tensordot/transpose	Transposedense_6/Relu:activations:0!dense_8/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????]??
dense_8/Tensordot/ReshapeReshapedense_8/Tensordot/transpose:y:0 dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_8/Tensordot/concat_1ConcatV2#dense_8/Tensordot/GatherV2:output:0"dense_8/Tensordot/Const_2:output:0(dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0#dense_8/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????]?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_8/BiasAddBiasAdddense_8/Tensordot:output:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]h
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????]c
conv2d_transpose_10/ShapeShapedense_8/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_10/strided_sliceStridedSlice"conv2d_transpose_10/Shape:output:00conv2d_transpose_10/strided_slice/stack:output:02conv2d_transpose_10/strided_slice/stack_1:output:02conv2d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_10/stack/1Const*
_output_shapes
: *
dtype0*
value	B :z]
conv2d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_10/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_10/stackPack*conv2d_transpose_10/strided_slice:output:0$conv2d_transpose_10/stack/1:output:0$conv2d_transpose_10/stack/2:output:0$conv2d_transpose_10/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_10/strided_slice_1StridedSlice"conv2d_transpose_10/stack:output:02conv2d_transpose_10/strided_slice_1/stack:output:04conv2d_transpose_10/strided_slice_1/stack_1:output:04conv2d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_10/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_10_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
$conv2d_transpose_10/conv2d_transposeConv2DBackpropInput"conv2d_transpose_10/stack:output:0;conv2d_transpose_10/conv2d_transpose/ReadVariableOp:value:0dense_8/Relu:activations:0*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
?
*conv2d_transpose_10/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_10/BiasAddBiasAdd-conv2d_transpose_10/conv2d_transpose:output:02conv2d_transpose_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????zm
conv2d_transpose_11/ShapeShape$conv2d_transpose_10/BiasAdd:output:0*
T0*
_output_shapes
:q
'conv2d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_11/strided_sliceStridedSlice"conv2d_transpose_11/Shape:output:00conv2d_transpose_11/strided_slice/stack:output:02conv2d_transpose_11/strided_slice/stack_1:output:02conv2d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_11/stack/1Const*
_output_shapes
: *
dtype0*
value	B :z^
conv2d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?]
conv2d_transpose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_11/stackPack*conv2d_transpose_11/strided_slice:output:0$conv2d_transpose_11/stack/1:output:0$conv2d_transpose_11/stack/2:output:0$conv2d_transpose_11/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_11/strided_slice_1StridedSlice"conv2d_transpose_11/stack:output:02conv2d_transpose_11/strided_slice_1/stack:output:04conv2d_transpose_11/strided_slice_1/stack_1:output:04conv2d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_11/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_11_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
$conv2d_transpose_11/conv2d_transposeConv2DBackpropInput"conv2d_transpose_11/stack:output:0;conv2d_transpose_11/conv2d_transpose/ReadVariableOp:value:0$conv2d_transpose_10/BiasAdd:output:0*
T0*0
_output_shapes
:?????????z?*
paddingVALID*
strides
?
*conv2d_transpose_11/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_11/BiasAddBiasAdd-conv2d_transpose_11/conv2d_transpose:output:02conv2d_transpose_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z??
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0`
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          a
dense_9/Tensordot/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:a
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_9/Tensordot/GatherV2_1GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/axes:output:0*dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_9/Tensordot/transpose	Transposedense_6/Relu:activations:0!dense_9/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????]??
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????]?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]h
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????]c
conv2d_transpose_12/ShapeShapedense_9/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_12/strided_sliceStridedSlice"conv2d_transpose_12/Shape:output:00conv2d_transpose_12/strided_slice/stack:output:02conv2d_transpose_12/strided_slice/stack_1:output:02conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :z]
conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_12/stackPack*conv2d_transpose_12/strided_slice:output:0$conv2d_transpose_12/stack/1:output:0$conv2d_transpose_12/stack/2:output:0$conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_12/strided_slice_1StridedSlice"conv2d_transpose_12/stack:output:02conv2d_transpose_12/strided_slice_1/stack:output:04conv2d_transpose_12/strided_slice_1/stack_1:output:04conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
$conv2d_transpose_12/conv2d_transposeConv2DBackpropInput"conv2d_transpose_12/stack:output:0;conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0dense_9/Relu:activations:0*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
?
*conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_12/BiasAddBiasAdd-conv2d_transpose_12/conv2d_transpose:output:02conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????zm
conv2d_transpose_13/ShapeShape$conv2d_transpose_12/BiasAdd:output:0*
T0*
_output_shapes
:q
'conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_13/strided_sliceStridedSlice"conv2d_transpose_13/Shape:output:00conv2d_transpose_13/strided_slice/stack:output:02conv2d_transpose_13/strided_slice/stack_1:output:02conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value	B :z^
conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?]
conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_13/stackPack*conv2d_transpose_13/strided_slice:output:0$conv2d_transpose_13/stack/1:output:0$conv2d_transpose_13/stack/2:output:0$conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_13/strided_slice_1StridedSlice"conv2d_transpose_13/stack:output:02conv2d_transpose_13/strided_slice_1/stack:output:04conv2d_transpose_13/strided_slice_1/stack_1:output:04conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_13_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
$conv2d_transpose_13/conv2d_transposeConv2DBackpropInput"conv2d_transpose_13/stack:output:0;conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:0$conv2d_transpose_12/BiasAdd:output:0*
T0*0
_output_shapes
:?????????z?*
paddingVALID*
strides
?
*conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_13/BiasAddBiasAdd-conv2d_transpose_13/conv2d_transpose:output:02conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z??
!dense_10/Tensordot/ReadVariableOpReadVariableOp*dense_10_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype0a
dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:l
dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          b
dense_10/Tensordot/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:b
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_10/Tensordot/GatherV2_1GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/axes:output:0+dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_10/Tensordot/transpose	Transposedense_6/Relu:activations:0"dense_10/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????]??
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????]?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????]j
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*/
_output_shapes
:?????????]d
conv2d_transpose_14/ShapeShapedense_10/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_14/strided_sliceStridedSlice"conv2d_transpose_14/Shape:output:00conv2d_transpose_14/strided_slice/stack:output:02conv2d_transpose_14/strided_slice/stack_1:output:02conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :z]
conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_14/stackPack*conv2d_transpose_14/strided_slice:output:0$conv2d_transpose_14/stack/1:output:0$conv2d_transpose_14/stack/2:output:0$conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_14/strided_slice_1StridedSlice"conv2d_transpose_14/stack:output:02conv2d_transpose_14/strided_slice_1/stack:output:04conv2d_transpose_14/strided_slice_1/stack_1:output:04conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
$conv2d_transpose_14/conv2d_transposeConv2DBackpropInput"conv2d_transpose_14/stack:output:0;conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:0dense_10/Relu:activations:0*
T0*/
_output_shapes
:?????????z*
paddingVALID*
strides
?
*conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_14/BiasAddBiasAdd-conv2d_transpose_14/conv2d_transpose:output:02conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????zm
conv2d_transpose_15/ShapeShape$conv2d_transpose_14/BiasAdd:output:0*
T0*
_output_shapes
:q
'conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_15/strided_sliceStridedSlice"conv2d_transpose_15/Shape:output:00conv2d_transpose_15/strided_slice/stack:output:02conv2d_transpose_15/strided_slice/stack_1:output:02conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :z^
conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?]
conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_15/stackPack*conv2d_transpose_15/strided_slice:output:0$conv2d_transpose_15/stack/1:output:0$conv2d_transpose_15/stack/2:output:0$conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_15/strided_slice_1StridedSlice"conv2d_transpose_15/stack:output:02conv2d_transpose_15/strided_slice_1/stack:output:04conv2d_transpose_15/strided_slice_1/stack_1:output:04conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_15_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
$conv2d_transpose_15/conv2d_transposeConv2DBackpropInput"conv2d_transpose_15/stack:output:0;conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0$conv2d_transpose_14/BiasAdd:output:0*
T0*0
_output_shapes
:?????????z?*
paddingVALID*
strides
?
*conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_15/BiasAddBiasAdd-conv2d_transpose_15/conv2d_transpose:output:02conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z?[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_1/concatConcatV2#conv2d_transpose_9/BiasAdd:output:0$conv2d_transpose_11/BiasAdd:output:0$conv2d_transpose_13/BiasAdd:output:0$conv2d_transpose_15/BiasAdd:output:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????z?x?
!dense_11/Tensordot/ReadVariableOpReadVariableOp*dense_11_tensordot_readvariableop_resource*
_output_shapes

:x*
dtype0a
dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:l
dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          e
dense_11/Tensordot/ShapeShapeconcatenate_1/concat:output:0*
T0*
_output_shapes
:b
 dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_11/Tensordot/GatherV2GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/free:output:0)dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_11/Tensordot/GatherV2_1GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/axes:output:0+dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_11/Tensordot/ProdProd$dense_11/Tensordot/GatherV2:output:0!dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_11/Tensordot/Prod_1Prod&dense_11/Tensordot/GatherV2_1:output:0#dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_11/Tensordot/concatConcatV2 dense_11/Tensordot/free:output:0 dense_11/Tensordot/axes:output:0'dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_11/Tensordot/stackPack dense_11/Tensordot/Prod:output:0"dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_11/Tensordot/transpose	Transposeconcatenate_1/concat:output:0"dense_11/Tensordot/concat:output:0*
T0*0
_output_shapes
:?????????z?x?
dense_11/Tensordot/ReshapeReshape dense_11/Tensordot/transpose:y:0!dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_11/Tensordot/MatMulMatMul#dense_11/Tensordot/Reshape:output:0)dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_11/Tensordot/concat_1ConcatV2$dense_11/Tensordot/GatherV2:output:0#dense_11/Tensordot/Const_2:output:0)dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_11/TensordotReshape#dense_11/Tensordot/MatMul:product:0$dense_11/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:?????????z??
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_11/BiasAddBiasAdddense_11/Tensordot:output:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z?k
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*0
_output_shapes
:?????????z?s
IdentityIdentitydense_11/Relu:activations:0^NoOp*
T0*0
_output_shapes
:?????????z??

NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp+^conv2d_transpose_10/BiasAdd/ReadVariableOp4^conv2d_transpose_10/conv2d_transpose/ReadVariableOp+^conv2d_transpose_11/BiasAdd/ReadVariableOp4^conv2d_transpose_11/conv2d_transpose/ReadVariableOp+^conv2d_transpose_12/BiasAdd/ReadVariableOp4^conv2d_transpose_12/conv2d_transpose/ReadVariableOp+^conv2d_transpose_13/BiasAdd/ReadVariableOp4^conv2d_transpose_13/conv2d_transpose/ReadVariableOp+^conv2d_transpose_14/BiasAdd/ReadVariableOp4^conv2d_transpose_14/conv2d_transpose/ReadVariableOp+^conv2d_transpose_15/BiasAdd/ReadVariableOp4^conv2d_transpose_15/conv2d_transpose/ReadVariableOp*^conv2d_transpose_8/BiasAdd/ReadVariableOp3^conv2d_transpose_8/conv2d_transpose/ReadVariableOp*^conv2d_transpose_9/BiasAdd/ReadVariableOp3^conv2d_transpose_9/conv2d_transpose/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp"^dense_11/Tensordot/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/Tensordot/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/Tensordot/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:?????????z?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2X
*conv2d_transpose_10/BiasAdd/ReadVariableOp*conv2d_transpose_10/BiasAdd/ReadVariableOp2j
3conv2d_transpose_10/conv2d_transpose/ReadVariableOp3conv2d_transpose_10/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_11/BiasAdd/ReadVariableOp*conv2d_transpose_11/BiasAdd/ReadVariableOp2j
3conv2d_transpose_11/conv2d_transpose/ReadVariableOp3conv2d_transpose_11/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_12/BiasAdd/ReadVariableOp*conv2d_transpose_12/BiasAdd/ReadVariableOp2j
3conv2d_transpose_12/conv2d_transpose/ReadVariableOp3conv2d_transpose_12/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_13/BiasAdd/ReadVariableOp*conv2d_transpose_13/BiasAdd/ReadVariableOp2j
3conv2d_transpose_13/conv2d_transpose/ReadVariableOp3conv2d_transpose_13/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_14/BiasAdd/ReadVariableOp*conv2d_transpose_14/BiasAdd/ReadVariableOp2j
3conv2d_transpose_14/conv2d_transpose/ReadVariableOp3conv2d_transpose_14/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_15/BiasAdd/ReadVariableOp*conv2d_transpose_15/BiasAdd/ReadVariableOp2j
3conv2d_transpose_15/conv2d_transpose/ReadVariableOp3conv2d_transpose_15/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_8/BiasAdd/ReadVariableOp)conv2d_transpose_8/BiasAdd/ReadVariableOp2h
2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_9/BiasAdd/ReadVariableOp)conv2d_transpose_9/BiasAdd/ReadVariableOp2h
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2F
!dense_11/Tensordot/ReadVariableOp!dense_11/Tensordot/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2D
 dense_6/Tensordot/ReadVariableOp dense_6/Tensordot/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp:S O
0
_output_shapes
:?????????z?

_user_specified_namex
?
?
Q__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_21060449

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
P__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_21061772

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
*__inference_dense_9_layer_call_fn_21061646

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
GPU 2J 8? *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_21060421w
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
?#
?
Q__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_21062084

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
?
$__inference__traced_restore_21062623
file_prefixU
:assignvariableop_source_separation_model_1_conv2d_2_kernel:?H
:assignvariableop_1_source_separation_model_1_conv2d_2_bias:V
<assignvariableop_2_source_separation_model_1_conv2d_3_kernel:H
:assignvariableop_3_source_separation_model_1_conv2d_3_bias:N
;assignvariableop_4_source_separation_model_1_dense_6_kernel:	?H
9assignvariableop_5_source_separation_model_1_dense_6_bias:	?N
;assignvariableop_6_source_separation_model_1_dense_7_kernel:	?G
9assignvariableop_7_source_separation_model_1_dense_7_bias:N
;assignvariableop_8_source_separation_model_1_dense_8_kernel:	?G
9assignvariableop_9_source_separation_model_1_dense_8_bias:O
<assignvariableop_10_source_separation_model_1_dense_9_kernel:	?H
:assignvariableop_11_source_separation_model_1_dense_9_bias:P
=assignvariableop_12_source_separation_model_1_dense_10_kernel:	?I
;assignvariableop_13_source_separation_model_1_dense_10_bias:a
Gassignvariableop_14_source_separation_model_1_conv2d_transpose_8_kernel:S
Eassignvariableop_15_source_separation_model_1_conv2d_transpose_8_bias:b
Gassignvariableop_16_source_separation_model_1_conv2d_transpose_9_kernel:?S
Eassignvariableop_17_source_separation_model_1_conv2d_transpose_9_bias:b
Hassignvariableop_18_source_separation_model_1_conv2d_transpose_10_kernel:T
Fassignvariableop_19_source_separation_model_1_conv2d_transpose_10_bias:c
Hassignvariableop_20_source_separation_model_1_conv2d_transpose_11_kernel:?T
Fassignvariableop_21_source_separation_model_1_conv2d_transpose_11_bias:b
Hassignvariableop_22_source_separation_model_1_conv2d_transpose_12_kernel:T
Fassignvariableop_23_source_separation_model_1_conv2d_transpose_12_bias:c
Hassignvariableop_24_source_separation_model_1_conv2d_transpose_13_kernel:?T
Fassignvariableop_25_source_separation_model_1_conv2d_transpose_13_bias:b
Hassignvariableop_26_source_separation_model_1_conv2d_transpose_14_kernel:T
Fassignvariableop_27_source_separation_model_1_conv2d_transpose_14_bias:c
Hassignvariableop_28_source_separation_model_1_conv2d_transpose_15_kernel:?T
Fassignvariableop_29_source_separation_model_1_conv2d_transpose_15_bias:O
=assignvariableop_30_source_separation_model_1_dense_11_kernel:xI
;assignvariableop_31_source_separation_model_1_dense_11_bias:
identity_33??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB)dClass0/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dClass0/bias/.ATTRIBUTES/VARIABLE_VALUEB)dClass1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dClass1/bias/.ATTRIBUTES/VARIABLE_VALUEB)dClass2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dClass2/bias/.ATTRIBUTES/VARIABLE_VALUEB)dClass3/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dClass3/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv3Class0/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv3Class0/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv4Class0/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv4Class0/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv3Class1/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv3Class1/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv4Class1/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv4Class1/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv3Class2/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv3Class2/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv4Class2/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv4Class2/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv3Class3/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv3Class3/bias/.ATTRIBUTES/VARIABLE_VALUEB-conv4Class3/kernel/.ATTRIBUTES/VARIABLE_VALUEB+conv4Class3/bias/.ATTRIBUTES/VARIABLE_VALUEB%out/kernel/.ATTRIBUTES/VARIABLE_VALUEB#out/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::*/
dtypes%
#2![
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp:assignvariableop_source_separation_model_1_conv2d_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp:assignvariableop_1_source_separation_model_1_conv2d_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp<assignvariableop_2_source_separation_model_1_conv2d_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp:assignvariableop_3_source_separation_model_1_conv2d_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp;assignvariableop_4_source_separation_model_1_dense_6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_source_separation_model_1_dense_6_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp;assignvariableop_6_source_separation_model_1_dense_7_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp9assignvariableop_7_source_separation_model_1_dense_7_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp;assignvariableop_8_source_separation_model_1_dense_8_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp9assignvariableop_9_source_separation_model_1_dense_8_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp<assignvariableop_10_source_separation_model_1_dense_9_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_source_separation_model_1_dense_9_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp=assignvariableop_12_source_separation_model_1_dense_10_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp;assignvariableop_13_source_separation_model_1_dense_10_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpGassignvariableop_14_source_separation_model_1_conv2d_transpose_8_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpEassignvariableop_15_source_separation_model_1_conv2d_transpose_8_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpGassignvariableop_16_source_separation_model_1_conv2d_transpose_9_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpEassignvariableop_17_source_separation_model_1_conv2d_transpose_9_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpHassignvariableop_18_source_separation_model_1_conv2d_transpose_10_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpFassignvariableop_19_source_separation_model_1_conv2d_transpose_10_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpHassignvariableop_20_source_separation_model_1_conv2d_transpose_11_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpFassignvariableop_21_source_separation_model_1_conv2d_transpose_11_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpHassignvariableop_22_source_separation_model_1_conv2d_transpose_12_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpFassignvariableop_23_source_separation_model_1_conv2d_transpose_12_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpHassignvariableop_24_source_separation_model_1_conv2d_transpose_13_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpFassignvariableop_25_source_separation_model_1_conv2d_transpose_13_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpHassignvariableop_26_source_separation_model_1_conv2d_transpose_14_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpFassignvariableop_27_source_separation_model_1_conv2d_transpose_14_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpHassignvariableop_28_source_separation_model_1_conv2d_transpose_15_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpFassignvariableop_29_source_separation_model_1_conv2d_transpose_15_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp=assignvariableop_30_source_separation_model_1_dense_11_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp;assignvariableop_31_source_separation_model_1_dense_11_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_32Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_33IdentityIdentity_32:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_33Identity_33:output:0*U
_input_shapesD
B: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312(
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
?
?
6__inference_conv2d_transpose_13_layer_call_fn_21062116

inputs"
unknown:?
	unknown_0:
identity??StatefulPartitionedCall?
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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_21060025?
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
F__inference_dense_10_layer_call_and_return_conditional_losses_21061717

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
?
?
<__inference_source_separation_model_1_layer_call_fn_21060692
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

unknown_17:	?

unknown_18:$

unknown_19:

unknown_20:%

unknown_21:?

unknown_22:

unknown_23:	?

unknown_24:$

unknown_25:

unknown_26:%

unknown_27:?

unknown_28:

unknown_29:x

unknown_30:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8? *`
f[RY
W__inference_source_separation_model_1_layer_call_and_return_conditional_losses_21060625x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????z?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:?????????z?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:?????????z?
!
_user_specified_name	input_1
?#
?
Q__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_21060073

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
?#
?
Q__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_21059929

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
?
?
Q__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_21060477

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
?#
?
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_21060121

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
K__inference_concatenate_1_layer_call_and_return_conditional_losses_21062358
inputs_0
inputs_1
inputs_2
inputs_3
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputs_0inputs_1inputs_2inputs_3concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????z?x`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????z?x"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesr
p:?????????z?:?????????z?:?????????z?:?????????z?:Z V
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
inputs/1:ZV
0
_output_shapes
:?????????z?
"
_user_specified_name
inputs/2:ZV
0
_output_shapes
:?????????z?
"
_user_specified_name
inputs/3
?
?
E__inference_dense_7_layer_call_and_return_conditional_losses_21061597

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
?
?
*__inference_dense_8_layer_call_fn_21061606

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
GPU 2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_21060328w
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
?
?
6__inference_conv2d_transpose_12_layer_call_fn_21062038

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_21059977?
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
P__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_21061850

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
?
?
P__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_21060263

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
?
?
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_21062341

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
E__inference_dense_6_layer_call_and_return_conditional_losses_21061557

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
?
?
E__inference_dense_8_layer_call_and_return_conditional_losses_21061637

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

?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_21060145

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
?#
?
Q__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_21061928

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
?
?
Q__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_21062107

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
6__inference_conv2d_transpose_11_layer_call_fn_21061960

inputs"
unknown:?
	unknown_0:
identity??StatefulPartitionedCall?
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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_21059929?
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
?
?
6__inference_conv2d_transpose_15_layer_call_fn_21062272

inputs"
unknown:?
	unknown_0:
identity??StatefulPartitionedCall?
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
GPU 2J 8? *Z
fURS
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_21060121?
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
?#
?
P__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_21059785

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
x
0__inference_concatenate_1_layer_call_fn_21062349
inputs_0
inputs_1
inputs_2
inputs_3
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????z?x* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concatenate_1_layer_call_and_return_conditional_losses_21060585i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????z?x"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesr
p:?????????z?:?????????z?:?????????z?:?????????z?:Z V
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
inputs/1:ZV
0
_output_shapes
:?????????z?
"
_user_specified_name
inputs/2:ZV
0
_output_shapes
:?????????z?
"
_user_specified_name
inputs/3
?
?
5__inference_conv2d_transpose_9_layer_call_fn_21061813

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
GPU 2J 8? *Y
fTRR
P__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_21060291x
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
?

?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_21060161

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
?
?
F__inference_dense_11_layer_call_and_return_conditional_losses_21060618

inputs3
!tensordot_readvariableop_resource:x-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:x*
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
:?????????z?x?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
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
:?????????z?r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????z?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????z?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????z?z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????z?x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:X T
0
_output_shapes
:?????????z?x
 
_user_specified_nameinputs
?
?
+__inference_dense_10_layer_call_fn_21061686

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
GPU 2J 8? *O
fJRH
F__inference_dense_10_layer_call_and_return_conditional_losses_21060514w
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
StatefulPartitionedCall:0?????????z?tensorflow/serving/predict:ܫ
?
	conv1
	conv2
d1
dClass0
dClass1
dClass2
dClass3
conv3Class0
	conv4Class0

conv3Class1
conv4Class1
conv3Class2
conv4Class2
conv3Class3
conv4Class3

concat
out
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_model
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

Gkernel
Hbias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Mkernel
Nbias
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

_kernel
`bias
a	variables
btrainable_variables
cregularization_losses
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

ekernel
fbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kkernel
lbias
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

ukernel
vbias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
0
1
2
3
#4
$5
)6
*7
/8
09
510
611
;12
<13
A14
B15
G16
H17
M18
N19
S20
T21
Y22
Z23
_24
`25
e26
f27
k28
l29
u30
v31"
trackable_list_wrapper
?
0
1
2
3
#4
$5
)6
*7
/8
09
510
611
;12
<13
A14
B15
G16
H17
M18
N19
S20
T21
Y22
Z23
_24
`25
e26
f27
k28
l29
u30
v31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
D:B?2)source_separation_model_1/conv2d_2/kernel
5:32'source_separation_model_1/conv2d_2/bias
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
C:A2)source_separation_model_1/conv2d_3/kernel
5:32'source_separation_model_1/conv2d_3/bias
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
 trainable_variables
!regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
;:9	?2(source_separation_model_1/dense_6/kernel
5:3?2&source_separation_model_1/dense_6/bias
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
%	variables
&trainable_variables
'regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
;:9	?2(source_separation_model_1/dense_7/kernel
4:22&source_separation_model_1/dense_7/bias
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
+	variables
,trainable_variables
-regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
;:9	?2(source_separation_model_1/dense_8/kernel
4:22&source_separation_model_1/dense_8/bias
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
1	variables
2trainable_variables
3regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
;:9	?2(source_separation_model_1/dense_9/kernel
4:22&source_separation_model_1/dense_9/bias
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
7	variables
8trainable_variables
9regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
<::	?2)source_separation_model_1/dense_10/kernel
5:32'source_separation_model_1/dense_10/bias
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
=	variables
>trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
M:K23source_separation_model_1/conv2d_transpose_8/kernel
?:=21source_separation_model_1/conv2d_transpose_8/bias
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
?non_trainable_variables
?layers
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
N:L?23source_separation_model_1/conv2d_transpose_9/kernel
?:=21source_separation_model_1/conv2d_transpose_9/bias
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
N:L24source_separation_model_1/conv2d_transpose_10/kernel
@:>22source_separation_model_1/conv2d_transpose_10/bias
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
O:M?24source_separation_model_1/conv2d_transpose_11/kernel
@:>22source_separation_model_1/conv2d_transpose_11/bias
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
N:L24source_separation_model_1/conv2d_transpose_12/kernel
@:>22source_separation_model_1/conv2d_transpose_12/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
[	variables
\trainable_variables
]regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
O:M?24source_separation_model_1/conv2d_transpose_13/kernel
@:>22source_separation_model_1/conv2d_transpose_13/bias
.
_0
`1"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
a	variables
btrainable_variables
cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
N:L24source_separation_model_1/conv2d_transpose_14/kernel
@:>22source_separation_model_1/conv2d_transpose_14/bias
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
g	variables
htrainable_variables
iregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
O:M?24source_separation_model_1/conv2d_transpose_15/kernel
@:>22source_separation_model_1/conv2d_transpose_15/bias
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
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
q	variables
rtrainable_variables
sregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
;:9x2)source_separation_model_1/dense_11/kernel
5:32'source_separation_model_1/dense_11/bias
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
w	variables
xtrainable_variables
yregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
?
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
11
12
13
14
15
16"
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
<__inference_source_separation_model_1_layer_call_fn_21060692
<__inference_source_separation_model_1_layer_call_fn_21061147?
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
W__inference_source_separation_model_1_layer_call_and_return_conditional_losses_21061479
W__inference_source_separation_model_1_layer_call_and_return_conditional_losses_21061007?
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
#__inference__wrapped_model_21059744input_1"?
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
+__inference_conv2d_2_layer_call_fn_21061488?
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
F__inference_conv2d_2_layer_call_and_return_conditional_losses_21061498?
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
+__inference_conv2d_3_layer_call_fn_21061507?
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
F__inference_conv2d_3_layer_call_and_return_conditional_losses_21061517?
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
*__inference_dense_6_layer_call_fn_21061526?
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
E__inference_dense_6_layer_call_and_return_conditional_losses_21061557?
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
*__inference_dense_7_layer_call_fn_21061566?
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
E__inference_dense_7_layer_call_and_return_conditional_losses_21061597?
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
*__inference_dense_8_layer_call_fn_21061606?
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
E__inference_dense_8_layer_call_and_return_conditional_losses_21061637?
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
*__inference_dense_9_layer_call_fn_21061646?
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
E__inference_dense_9_layer_call_and_return_conditional_losses_21061677?
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
+__inference_dense_10_layer_call_fn_21061686?
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
F__inference_dense_10_layer_call_and_return_conditional_losses_21061717?
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
5__inference_conv2d_transpose_8_layer_call_fn_21061726
5__inference_conv2d_transpose_8_layer_call_fn_21061735?
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
P__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_21061772
P__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_21061795?
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
5__inference_conv2d_transpose_9_layer_call_fn_21061804
5__inference_conv2d_transpose_9_layer_call_fn_21061813?
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
P__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_21061850
P__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_21061873?
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
6__inference_conv2d_transpose_10_layer_call_fn_21061882
6__inference_conv2d_transpose_10_layer_call_fn_21061891?
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
Q__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_21061928
Q__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_21061951?
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
6__inference_conv2d_transpose_11_layer_call_fn_21061960
6__inference_conv2d_transpose_11_layer_call_fn_21061969?
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
Q__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_21062006
Q__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_21062029?
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
6__inference_conv2d_transpose_12_layer_call_fn_21062038
6__inference_conv2d_transpose_12_layer_call_fn_21062047?
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
Q__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_21062084
Q__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_21062107?
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
6__inference_conv2d_transpose_13_layer_call_fn_21062116
6__inference_conv2d_transpose_13_layer_call_fn_21062125?
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
Q__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_21062162
Q__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_21062185?
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
6__inference_conv2d_transpose_14_layer_call_fn_21062194
6__inference_conv2d_transpose_14_layer_call_fn_21062203?
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
Q__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_21062240
Q__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_21062263?
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
6__inference_conv2d_transpose_15_layer_call_fn_21062272
6__inference_conv2d_transpose_15_layer_call_fn_21062281?
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
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_21062318
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_21062341?
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
0__inference_concatenate_1_layer_call_fn_21062349?
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
K__inference_concatenate_1_layer_call_and_return_conditional_losses_21062358?
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
+__inference_dense_11_layer_call_fn_21062367?
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
F__inference_dense_11_layer_call_and_return_conditional_losses_21062398?
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
&__inference_signature_wrapper_21061078input_1"?
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
#__inference__wrapped_model_21059744? #$)*ABGH/0MNST56YZ_`;<efkluv9?6
/?,
*?'
input_1?????????z?
? "<?9
7
output_1+?(
output_1?????????z??
K__inference_concatenate_1_layer_call_and_return_conditional_losses_21062358????
???
???
+?(
inputs/0?????????z?
+?(
inputs/1?????????z?
+?(
inputs/2?????????z?
+?(
inputs/3?????????z?
? ".?+
$?!
0?????????z?x
? ?
0__inference_concatenate_1_layer_call_fn_21062349????
???
???
+?(
inputs/0?????????z?
+?(
inputs/1?????????z?
+?(
inputs/2?????????z?
+?(
inputs/3?????????z?
? "!??????????z?x?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_21061498m8?5
.?+
)?&
inputs?????????z?
? "-?*
#? 
0?????????z
? ?
+__inference_conv2d_2_layer_call_fn_21061488`8?5
.?+
)?&
inputs?????????z?
? " ??????????z?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_21061517l7?4
-?*
(?%
inputs?????????z
? "-?*
#? 
0?????????]
? ?
+__inference_conv2d_3_layer_call_fn_21061507_7?4
-?*
(?%
inputs?????????z
? " ??????????]?
Q__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_21061928?MNI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_21061951lMN7?4
-?*
(?%
inputs?????????]
? "-?*
#? 
0?????????z
? ?
6__inference_conv2d_transpose_10_layer_call_fn_21061882?MNI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
6__inference_conv2d_transpose_10_layer_call_fn_21061891_MN7?4
-?*
(?%
inputs?????????]
? " ??????????z?
Q__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_21062006?STI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_21062029mST7?4
-?*
(?%
inputs?????????z
? ".?+
$?!
0?????????z?
? ?
6__inference_conv2d_transpose_11_layer_call_fn_21061960?STI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
6__inference_conv2d_transpose_11_layer_call_fn_21061969`ST7?4
-?*
(?%
inputs?????????z
? "!??????????z??
Q__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_21062084?YZI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_21062107lYZ7?4
-?*
(?%
inputs?????????]
? "-?*
#? 
0?????????z
? ?
6__inference_conv2d_transpose_12_layer_call_fn_21062038?YZI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
6__inference_conv2d_transpose_12_layer_call_fn_21062047_YZ7?4
-?*
(?%
inputs?????????]
? " ??????????z?
Q__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_21062162?_`I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_21062185m_`7?4
-?*
(?%
inputs?????????z
? ".?+
$?!
0?????????z?
? ?
6__inference_conv2d_transpose_13_layer_call_fn_21062116?_`I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
6__inference_conv2d_transpose_13_layer_call_fn_21062125`_`7?4
-?*
(?%
inputs?????????z
? "!??????????z??
Q__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_21062240?efI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_21062263lef7?4
-?*
(?%
inputs?????????]
? "-?*
#? 
0?????????z
? ?
6__inference_conv2d_transpose_14_layer_call_fn_21062194?efI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
6__inference_conv2d_transpose_14_layer_call_fn_21062203_ef7?4
-?*
(?%
inputs?????????]
? " ??????????z?
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_21062318?klI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_21062341mkl7?4
-?*
(?%
inputs?????????z
? ".?+
$?!
0?????????z?
? ?
6__inference_conv2d_transpose_15_layer_call_fn_21062272?klI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
6__inference_conv2d_transpose_15_layer_call_fn_21062281`kl7?4
-?*
(?%
inputs?????????z
? "!??????????z??
P__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_21061772?ABI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
P__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_21061795lAB7?4
-?*
(?%
inputs?????????]
? "-?*
#? 
0?????????z
? ?
5__inference_conv2d_transpose_8_layer_call_fn_21061726?ABI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
5__inference_conv2d_transpose_8_layer_call_fn_21061735_AB7?4
-?*
(?%
inputs?????????]
? " ??????????z?
P__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_21061850?GHI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
P__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_21061873mGH7?4
-?*
(?%
inputs?????????z
? ".?+
$?!
0?????????z?
? ?
5__inference_conv2d_transpose_9_layer_call_fn_21061804?GHI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
5__inference_conv2d_transpose_9_layer_call_fn_21061813`GH7?4
-?*
(?%
inputs?????????z
? "!??????????z??
F__inference_dense_10_layer_call_and_return_conditional_losses_21061717m;<8?5
.?+
)?&
inputs?????????]?
? "-?*
#? 
0?????????]
? ?
+__inference_dense_10_layer_call_fn_21061686`;<8?5
.?+
)?&
inputs?????????]?
? " ??????????]?
F__inference_dense_11_layer_call_and_return_conditional_losses_21062398nuv8?5
.?+
)?&
inputs?????????z?x
? ".?+
$?!
0?????????z?
? ?
+__inference_dense_11_layer_call_fn_21062367auv8?5
.?+
)?&
inputs?????????z?x
? "!??????????z??
E__inference_dense_6_layer_call_and_return_conditional_losses_21061557m#$7?4
-?*
(?%
inputs?????????]
? ".?+
$?!
0?????????]?
? ?
*__inference_dense_6_layer_call_fn_21061526`#$7?4
-?*
(?%
inputs?????????]
? "!??????????]??
E__inference_dense_7_layer_call_and_return_conditional_losses_21061597m)*8?5
.?+
)?&
inputs?????????]?
? "-?*
#? 
0?????????]
? ?
*__inference_dense_7_layer_call_fn_21061566`)*8?5
.?+
)?&
inputs?????????]?
? " ??????????]?
E__inference_dense_8_layer_call_and_return_conditional_losses_21061637m/08?5
.?+
)?&
inputs?????????]?
? "-?*
#? 
0?????????]
? ?
*__inference_dense_8_layer_call_fn_21061606`/08?5
.?+
)?&
inputs?????????]?
? " ??????????]?
E__inference_dense_9_layer_call_and_return_conditional_losses_21061677m568?5
.?+
)?&
inputs?????????]?
? "-?*
#? 
0?????????]
? ?
*__inference_dense_9_layer_call_fn_21061646`568?5
.?+
)?&
inputs?????????]?
? " ??????????]?
&__inference_signature_wrapper_21061078? #$)*ABGH/0MNST56YZ_`;<efkluvD?A
? 
:?7
5
input_1*?'
input_1?????????z?"<?9
7
output_1+?(
output_1?????????z??
W__inference_source_separation_model_1_layer_call_and_return_conditional_losses_21061007? #$)*ABGH/0MNST56YZ_`;<efkluv9?6
/?,
*?'
input_1?????????z?
? ".?+
$?!
0?????????z?
? ?
W__inference_source_separation_model_1_layer_call_and_return_conditional_losses_21061479? #$)*ABGH/0MNST56YZ_`;<efkluv3?0
)?&
$?!
x?????????z?
? ".?+
$?!
0?????????z?
? ?
<__inference_source_separation_model_1_layer_call_fn_21060692? #$)*ABGH/0MNST56YZ_`;<efkluv9?6
/?,
*?'
input_1?????????z?
? "!??????????z??
<__inference_source_separation_model_1_layer_call_fn_21061147z #$)*ABGH/0MNST56YZ_`;<efkluv3?0
)?&
$?!
x?????????z?
? "!??????????z?