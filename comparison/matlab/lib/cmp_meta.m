% Provenance record of a Spinach comparison run: meta=cmp_meta(case_name,options).
% Records the Spinach commit and dirty state, the MATLAB version, the
% machine, the parallel pool size, the GPU switch and device, and the
% case options.
function meta=cmp_meta(case_name,options)
spinach_root=fileparts(fileparts(which('create')));
[status,commit]=system(['git -C "' spinach_root '" rev-parse HEAD']);
if status~=0, commit='unknown'; end
[~,describe]=system(['git -C "' spinach_root '" describe --tags --always --dirty']);
[~,porcelain]=system(['git -C "' spinach_root '" status --porcelain']);
status_lines=splitlines(strtrim(porcelain)); status_lines(cellfun(@isempty,status_lines))=[];
dirty=any((~startsWith(status_lines,'??'))|endsWith(strtrim(status_lines),'.m'));
pool=gcp('nocreate');
if isempty(pool), workers=0; else, workers=pool.NumWorkers; end
meta.side='spinach';
meta.case=case_name;
meta.created_utc=char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
meta.spinach_root=spinach_root;
meta.spinach_commit=strtrim(commit);
meta.spinach_describe=strtrim(describe);
meta.spinach_dirty=dirty;
meta.matlab_version=version();
meta.platform=computer();
[~,host]=system('hostname'); meta.hostname=strtrim(host);
try, meta.cores=feature('numcores'); catch, meta.cores=maxNumCompThreads(); end
meta.pool_workers=workers;
meta.gpu=cmp_option(options,'gpu',false);
meta.gpu_device='';
if meta.gpu
    try
        if gpuDeviceCount>0, d=gpuDevice(); meta.gpu_device=d.Name; else, meta.gpu_device='none available (Spinach falls back to CPU)'; end
    catch
        meta.gpu_device='none available (no Parallel Computing Toolbox GPU support)';
    end
end
meta.options=options;
end
