create table if not exists model_apps (
  id uuid primary key default gen_random_uuid(),
  owner_user_id text not null references "user"(id) on delete cascade,
  slug text not null check (slug ~ '^[a-z0-9][a-z0-9-]{0,62}$'),
  name text not null check (length(name) between 1 and 100),
  description text not null default '' check (length(description) <= 1000),
  published boolean not null default false,
  active_version integer,
  next_version integer not null default 1,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique(owner_user_id,slug)
);
create index if not exists model_apps_owner on model_apps(owner_user_id,created_at desc);

create table if not exists model_app_releases (
  id uuid primary key default gen_random_uuid(),
  app_id uuid not null references model_apps(id) on delete cascade,
  version integer not null,
  run_id uuid not null references training_tasks(id) on delete cascade,
  checkpoint_path text not null,
  checkpoint_sha256 text not null,
  model_type text not null,
  task_type text not null,
  config jsonb not null,
  tested_at timestamptz,
  created_at timestamptz not null default now(),
  unique(app_id,version)
);
create index if not exists model_app_releases_run on model_app_releases(run_id);

alter table compute_jobs add column if not exists is_model_app boolean not null default false;
alter table compute_jobs add column if not exists model_app_id uuid references model_apps(id) on delete set null;
alter table compute_jobs add column if not exists model_app_release_id uuid references model_app_releases(id) on delete set null;
alter table compute_jobs add column if not exists app_requester_user_id text references "user"(id) on delete set null;
create index if not exists compute_jobs_app on compute_jobs(model_app_id,created_at desc) where is_model_app;
create index if not exists compute_jobs_app_requester on compute_jobs(app_requester_user_id,created_at desc) where is_model_app;

-- A separate trigger survives repeated application of the existing compute migrations.
create or replace function guard_model_app_run() returns trigger language plpgsql as $$
begin
  if exists(select 1 from model_app_releases where run_id=old.id) then
    raise exception 'Delete the Model App before deleting a referenced run.' using errcode='55006';
  end if;
  return old;
end $$;
drop trigger if exists guard_model_app_run on training_tasks;
create trigger guard_model_app_run before delete on training_tasks for each row execute function guard_model_app_run();
