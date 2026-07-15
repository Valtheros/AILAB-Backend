create extension if not exists pgcrypto;

create table if not exists workspaces (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  slug text not null unique,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists workspace_members (
  workspace_id uuid not null references workspaces(id) on delete cascade,
  user_id text not null,
  role text not null check (role in ('owner', 'admin', 'member', 'viewer')),
  created_at timestamptz not null default now(),
  primary key (workspace_id, user_id)
);

create table if not exists projects (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references workspaces(id) on delete cascade,
  name text not null,
  slug text not null,
  task_type text,
  created_by text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (workspace_id, slug)
);

create table if not exists datasets (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references workspaces(id) on delete cascade,
  project_id uuid references projects(id) on delete set null,
  name text not null,
  slug text not null,
  storage_path text not null,
  formats jsonb not null default '[]'::jsonb,
  tasks jsonb not null default '[]'::jsonb,
  classes jsonb not null default '[]'::jsonb,
  created_by text not null,
  created_at timestamptz not null default now(),
  unique (workspace_id, slug)
);

alter table datasets alter column workspace_id drop not null;
alter table datasets add column if not exists owner_user_id text;
alter table datasets add column if not exists status text not null default 'active';
alter table datasets add column if not exists metadata jsonb not null default '{}'::jsonb;
alter table datasets add column if not exists updated_at timestamptz not null default now();

create table if not exists training_runs (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references workspaces(id) on delete cascade,
  project_id uuid references projects(id) on delete set null,
  dataset_id uuid references datasets(id) on delete set null,
  rq_job_id text unique,
  run_slug text not null,
  task_type text not null,
  model_type text not null,
  model_name text,
  params jsonb not null default '{}'::jsonb,
  status text not null default 'queued',
  storage_path text not null,
  latest_metrics jsonb,
  created_by text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  finished_at timestamptz,
  unique (workspace_id, run_slug)
);

alter table training_runs alter column workspace_id drop not null;
alter table training_runs add column if not exists owner_user_id text;
alter table training_runs add column if not exists dataset_slug text;
alter table training_runs add column if not exists error_detail text;

create table if not exists artifacts (
  id uuid primary key default gen_random_uuid(),
  training_run_id uuid not null references training_runs(id) on delete cascade,
  kind text not null,
  file_name text not null,
  storage_path text not null,
  content_type text,
  size_bytes bigint,
  created_at timestamptz not null default now()
);

create index if not exists idx_workspace_members_user_id on workspace_members(user_id);
create index if not exists idx_projects_workspace_id on projects(workspace_id);
create index if not exists idx_datasets_workspace_id on datasets(workspace_id);
create index if not exists idx_training_runs_workspace_id on training_runs(workspace_id);
create index if not exists idx_training_runs_created_by on training_runs(created_by);
create unique index if not exists idx_datasets_owner_slug on datasets(owner_user_id, slug) where owner_user_id is not null;
create unique index if not exists idx_runs_owner_slug on training_runs(owner_user_id, run_slug) where owner_user_id is not null;
create index if not exists idx_runs_dataset_status on training_runs(dataset_id, status);

do $$
begin
  if to_regclass('"user"') is not null and not exists (select 1 from pg_constraint where conname = 'datasets_owner_user_fk') then
    alter table datasets add constraint datasets_owner_user_fk foreign key (owner_user_id)
      references "user"(id) on delete restrict not valid;
  end if;
  if to_regclass('"user"') is not null and not exists (select 1 from pg_constraint where conname = 'training_runs_owner_user_fk') then
    alter table training_runs add constraint training_runs_owner_user_fk foreign key (owner_user_id)
      references "user"(id) on delete restrict not valid;
  end if;
end $$;

do $$
begin
  if exists (select 1 from pg_constraint where conname = 'datasets_owner_user_fk' and not convalidated) then
    alter table datasets validate constraint datasets_owner_user_fk;
  end if;
  if exists (select 1 from pg_constraint where conname = 'training_runs_owner_user_fk' and not convalidated) then
    alter table training_runs validate constraint training_runs_owner_user_fk;
  end if;
end $$;
