% Writes a numeric array, a sparse matrix, or a cell array thereof into
% an HDF5 fixture file using the layout read by respinach.oracle. Syntax:
%
%                  oracle_write(file_name,var_name,value)
%
% Parameters:
%
%    file_name - name of the HDF5 file, created if it does not exist
%
%    var_name  - HDF5 group path of the variable, e.g. '/ops/Lz'; the
%                group must not already exist
%
%    value     - numeric or logical array of any dimension, dense or
%                sparse, real or complex, or a cell array of such
%
% Outputs:
%
%    this function writes to the HDF5 file
%
% Note: each variable is an HDF5 group with attributes 'kind' ('dense',
%       'sparse', or 'cell'), 'shape' (Matlab size), and, for dense and
%       sparse kinds, 'complex' (1 if the Matlab array is complex, 0 if
%       it is real). Dense arrays are stored in 're' and, if complex,
%       'im' datasets, permuted so that the HDF5 C-order shape is equal
%       to the Matlab shape. Sparse matrices are stored as 0-based 'row'
%       and 'col' int64 index vectors and 're' and, if complex, 'im' va-
%       lue vectors, all of size [1 nnz] as seen from Python; the 'nnz'
%       attribute holds the number of stored elements. Datasets are not
%       written for empty arrays and for sparse matrices with nnz=0. Cell
%       array elements are stored as member variables named by their 0-
%       based column-major linear index. Logical and integer arrays are
%       converted to double precision.

function oracle_write(file_name,var_name,value)

% Check consistency
grumble(file_name,var_name,value);

% Create the file if it does not exist
if ~isfile(file_name)
    file_id=H5F.create(file_name,'H5F_ACC_EXCL','H5P_DEFAULT','H5P_DEFAULT');
    H5F.close(file_id);
end

% Create the variable group with any missing parent groups
file_id=H5F.open(file_name,'H5F_ACC_RDWR','H5P_DEFAULT');
link_props=H5P.create('H5P_LINK_CREATE');
H5P.set_create_intermediate_group(link_props,1);
group_id=H5G.create(file_id,var_name,link_props,'H5P_DEFAULT','H5P_DEFAULT');
H5G.close(group_id); H5P.close(link_props); H5F.close(file_id);

% Record the Matlab shape
h5writeatt(file_name,var_name,'shape',int64(size(value)));

% Decide how to proceed
if iscell(value)

    % Write cell array elements as member variables
    h5writeatt(file_name,var_name,'kind','cell');
    for n=1:numel(value)
        oracle_write(file_name,[var_name '/' int2str(n-1)],value{n});
    end

elseif issparse(value)

    % Extract 0-based triplets in column-major order
    h5writeatt(file_name,var_name,'kind','sparse');
    h5writeatt(file_name,var_name,'complex',int64(~isreal(value)));
    [rows,cols,vals]=find(double(value));
    h5writeatt(file_name,var_name,'nnz',int64(numel(vals)));

    % Write the triplets
    if ~isempty(vals)
        h5create(file_name,[var_name '/row'],[numel(rows) 1],'Datatype','int64');
        h5write(file_name,[var_name '/row'],int64(rows(:)-1));
        h5create(file_name,[var_name '/col'],[numel(cols) 1],'Datatype','int64');
        h5write(file_name,[var_name '/col'],int64(cols(:)-1));
        h5create(file_name,[var_name '/re'],[numel(vals) 1]);
        h5write(file_name,[var_name '/re'],real(vals(:)));
        if ~isreal(value)
            h5create(file_name,[var_name '/im'],[numel(vals) 1]);
            h5write(file_name,[var_name '/im'],imag(vals(:)));
        end
    end

else

    % Reverse the dimension order to compensate for HDF5 row-major storage
    h5writeatt(file_name,var_name,'kind','dense');
    h5writeatt(file_name,var_name,'complex',int64(~isreal(value)));
    data=permute(double(value),ndims(value):-1:1);

    % Write the real and imaginary parts
    if ~isempty(data)
        h5create(file_name,[var_name '/re'],size(data));
        h5write(file_name,[var_name '/re'],real(data));
        if ~isreal(value)
            h5create(file_name,[var_name '/im'],size(data));
            h5write(file_name,[var_name '/im'],imag(data));
        end
    end

end

end

% Consistency enforcement
function grumble(file_name,var_name,value)
if ~ischar(file_name)
    error('file_name must be a character string.');
end
if (~ischar(var_name))||isempty(var_name)||(var_name(1)~='/')
    error('var_name must be a character string starting with /');
end
if (~isnumeric(value))&&(~islogical(value))&&(~iscell(value))
    error('value must be a numeric array, a logical array, or a cell array.');
end
end

