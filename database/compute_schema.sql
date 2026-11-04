create table if not exists compute_nodes (
  id text primary key,
  ram_total_mb bigint not null default 0,
  ram_available_mb bigint not null default 0,
  cpu_count integer not null default 1,
  ram_budget_percent integer not null default 70 check (ram_budget_percent between 10 and 90),
  inference_burst integer not null default 0,
  cpu_worker_heartbeat timestamptz,
  cpu_worker_id text,
  cpu_health_error text,
  worker_ram_mb integer not null default 8192,
  worker_cpus integer not null default 4,
  heartbeat_at timestamptz not null default now()
);
create table if not exists gpu_devices (
  uuid text primary key,
  node_id text not null references compute_nodes(id),
  display_index integer not null,
  name text not null,
  total_mb bigint not null,
  free_mb bigint not null default 0,
  utilization integer,
  temperature integer,
  process_count integer,
  telemetry_ok boolean not null default false,
  enabled boolean not null default true,
  draining boolean not null default false,
  worker_id text,
  worker_heartbeat timestamptz,
  health_error text,
  observed_at timestamptz not null default now()
);
create table if not exists compute_user_quotas (
  owner_user_id text primary key references "user"(id) on delete cascade,
  max_gpu_jobs integer not null default 1 check (max_gpu_jobs between 1 and 32),
  last_dispatched_at timestamptz
);
alter table compute_user_quotas add column if not exists accepting_jobs boolean not null default true;
create table if not exists compute_jobs (
  id uuid primary key,
  owner_user_id text not null references "user"(id) on delete restrict,
  run_id uuid not null references training_tasks(id) on delete cascade,
  kind text not null check (kind in ('train','predict')),
  mode text not null check (mode in ('auto','gpu','cpu')),
  requested_gpu_uuid text,
  assigned_gpu_uuid text references gpu_devices(uuid),
  node_id text references compute_nodes(id),
  status text not null default 'queued' check (status in ('queued','dispatching','running','stopping','recovery_pending','completed','failed','cancelled')),
  payload jsonb not null,
  idempotency_key text,
  dispatch_token uuid,
  rq_job_id text,
  worker_id text,
  worker_ram_mb integer not null default 0,
  worker_cpus integer not null default 0,
  estimated_vram_mb integer not null default 0,
  queue_reason text,
  cancel_requested boolean not null default false,
  error_code text,
  error_message text,
  result_path text,
  created_at timestamptz not null default now(),
  dispatched_at timestamptz,
  started_at timestamptz,
  finished_at timestamptz,
  heartbeat_at timestamptz,
  cleaned_at timestamptz
);
create unique index if not exists compute_gpu_exclusive on compute_jobs(assigned_gpu_uuid)
  where assigned_gpu_uuid is not null and status in ('dispatching','running','stopping','recovery_pending');
create unique index if not exists compute_cpu_exclusive on compute_jobs(node_id)
  where mode='cpu' and node_id is not null and status in ('dispatching','running','stopping','recovery_pending');
create unique index if not exists compute_submission_unique on compute_jobs(owner_user_id,idempotency_key)
  where idempotency_key is not null;
create index if not exists compute_jobs_pending on compute_jobs(status,created_at);
create index if not exists compute_jobs_owner on compute_jobs(owner_user_id,created_at desc);
create index if not exists compute_jobs_run on compute_jobs(run_id,status);
alter table compute_jobs add column if not exists worker_cgroup text;
alter table training_tasks add column if not exists execution jsonb;
alter table training_tasks add column if not exists compute_job_id uuid references compute_jobs(id);

create or replace function guard_active_compute_run() returns trigger language plpgsql as $$
begin
  if exists(select 1 from compute_jobs where run_id=old.id and status in ('queued','dispatching','running','stopping','recovery_pending')) then
    raise exception 'Cancel active compute jobs before deleting or transferring this run.' using errcode='55006';
  end if;
  return old;
end $$;
drop trigger if exists guard_active_compute_run_delete on training_tasks;
create trigger guard_active_compute_run_delete before delete on training_tasks
  for each row execute function guard_active_compute_run();

-- Only drafts receive a selection. Never invent GPU attribution for old completed runs.
update training_tasks set execution=case
  when coalesce(params->>'_device_selection','auto')='auto' then '{"mode":"auto","gpuUuid":null}'::jsonb
  when params->>'device'='cpu' then '{"mode":"cpu","gpuUuid":null}'::jsonb
  else jsonb_build_object('mode','gpu','gpuUuid',null,'needsSelection',true)
end where status='draft' and execution is null;
