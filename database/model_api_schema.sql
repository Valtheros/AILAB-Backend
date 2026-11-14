create table if not exists model_endpoints (
  id uuid primary key default gen_random_uuid(),
  owner_user_id text not null references "user"(id) on delete cascade,
  run_id uuid not null references training_tasks(id) on delete cascade,
  checkpoint_path text not null,
  checkpoint_sha256 text not null,
  enabled boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique(run_id)
);
create index if not exists model_endpoints_owner on model_endpoints(owner_user_id,created_at desc);

create table if not exists model_api_keys (
  id uuid primary key default gen_random_uuid(),
  endpoint_id uuid not null references model_endpoints(id) on delete cascade,
  key_sha256 text not null unique,
  label text not null default '',
  created_at timestamptz not null default now(),
  expires_at timestamptz not null default now()+interval '90 days',
  revoked_at timestamptz
);
create index if not exists model_api_keys_endpoint on model_api_keys(endpoint_id,created_at desc);

alter table compute_jobs add column if not exists model_endpoint_id uuid references model_endpoints(id) on delete set null;
alter table compute_jobs add column if not exists model_api_key_id uuid references model_api_keys(id) on delete set null;
create index if not exists compute_jobs_api_key on compute_jobs(model_api_key_id,created_at desc);

create or replace function guard_active_compute_run() returns trigger language plpgsql as $$
begin
  if exists(select 1 from compute_jobs where run_id=old.id and status in ('queued','dispatching','running','stopping','recovery_pending'))
     or exists(select 1 from evaluations where run_id=old.id and status in ('awaiting_queue','queued','running','stopping'))
     or exists(select 1 from model_endpoints where run_id=old.id and enabled) then
    raise exception 'Disable the model API and cancel active jobs before deleting this run.' using errcode='55006';
  end if;
  return old;
end $$;
