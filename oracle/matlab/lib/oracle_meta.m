% Writes the meta.json provenance record of a fixture case. Records the
% Spinach commit, Matlab version, formalism, active tolerances, input
% parameters, SHA-256 hashes of the dump script and harness files, and
% any active shadow files from oracle/matlab/shadow. Syntax:
%
%             oracle_meta(case_dir,dump_file,spin_system,inputs)
%
% Parameters:
%
%    case_dir    - fixture case directory, must already exist and
%                  must not contain a meta.json file
%
%    dump_file   - full path to the .m file of the dump script
%
%    spin_system - Spinach spin system object after basis() call
%
%    inputs      - structure with the user-level inputs of the case
%                  (sys, inter, bas, and scalar sequence parameters)
%                  that jsonencode() can serialise
%
% Outputs:
%
%    this function writes case_dir/meta.json
%
% Note: the function stops with an error if the Spinach root directory
%       is not a git working tree, since the record would then not
%       identify the code that produced the fixture.

function oracle_meta(case_dir,dump_file,spin_system,inputs)

% Check consistency
grumble(case_dir,dump_file,spin_system,inputs);

% Locate Spinach and harness directories
spinach_root=fileparts(fileparts(which('create')));
lib_dir=fileparts(mfilename('fullpath'));
shadow_dir=fullfile(fileparts(lib_dir),'shadow');

% Identify the Spinach commit
[status,commit]=system(['git -C "' spinach_root '" rev-parse HEAD']);
if status~=0
    error(['Spinach at ' spinach_root ' is not a git working tree: ' commit]);
end
[~,describe]=system(['git -C "' spinach_root '" describe --tags --always --dirty']);
[~,porcelain]=system(['git -C "' spinach_root '" status --porcelain']);

% Dirty means modified tracked files or untracked .m files; untracked
% non-code files such as Finder's .DS_Store cannot change results
status_lines=splitlines(strtrim(porcelain)); status_lines(cellfun(@isempty,status_lines))=[];
untracked=startsWith(status_lines,'??');
relevant=(~untracked)|endsWith(strtrim(status_lines),'.m');

% Identify active shadow files
shadow_files=dir(fullfile(shadow_dir,'**','*.m'));
shadows=struct('name',{},'path',{},'sha256',{});
for n=1:numel(shadow_files)
    file_path=fullfile(shadow_files(n).folder,shadow_files(n).name);
    [~,func_name]=fileparts(file_path);
    if strcmp(which(func_name),file_path)
        shadows(end+1)=struct('name',func_name,'path',file_path,...
                              'sha256',file_sha256(file_path)); %#ok<AGROW>
    end
end

% Assemble the record
meta.schema_version=1;
meta.created_utc=char(datetime('now','TimeZone','UTC',...
                               'Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
meta.spinach_root=spinach_root;
meta.spinach_commit=strtrim(commit);
meta.spinach_describe=strtrim(describe);
meta.spinach_dirty=any(relevant);
meta.spinach_status=status_lines;
meta.matlab_version=version();
meta.formalism=spin_system.bas.formalism;
meta.approximation=spin_system.bas.approximation;
meta.tols=spin_system.tols;
meta.sys_enable=spin_system.sys.enable;
meta.sys_disable=spin_system.sys.disable;
meta.dump_script=struct('path',dump_file,'sha256',file_sha256(dump_file));
meta.harness=struct('oracle_write',file_sha256(fullfile(lib_dir,'oracle_write.m')),...
                    'oracle_meta',file_sha256(fullfile(lib_dir,'oracle_meta.m')));
meta.shadows=shadows;
meta.inputs=inputs;

% Write the record
file_id=fopen(fullfile(case_dir,'meta.json'),'w');
fprintf(file_id,'%s\n',jsonencode(meta,'PrettyPrint',true));
fclose(file_id);

end

% SHA-256 hex digest of a file
function hex_digest=file_sha256(file_path)
file_id=fopen(file_path,'r');
file_bytes=fread(file_id,Inf,'*uint8');
fclose(file_id);
hasher=java.security.MessageDigest.getInstance('SHA-256');
digest=typecast(hasher.digest(file_bytes),'uint8');
hex_digest=lower(reshape(dec2hex(digest,2).',1,[]));
end

% Consistency enforcement
function grumble(case_dir,dump_file,spin_system,inputs)
if (~ischar(case_dir))||(~isfolder(case_dir))
    error('case_dir must be the name of an existing directory.');
end
if isfile(fullfile(case_dir,'meta.json'))
    error('meta.json already exists in case_dir, fixtures are never overwritten.');
end
if (~ischar(dump_file))||(~isfile(dump_file))
    error('dump_file must be the name of an existing file.');
end
if (~isstruct(spin_system))||(~isfield(spin_system,'bas'))
    error('spin_system must be a Spinach spin system object after basis() call.');
end
if ~isstruct(inputs)
    error('inputs must be a structure.');
end
end

