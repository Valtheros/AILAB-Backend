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

do $$
begin
  if to_regclass('training_tasks') is null and to_regclass('training_runs') is not null then
    alter table training_runs rename to training_tasks;
  end if;
end $$;

create table if not exists training_tasks (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references workspaces(id) on delete cascade,
  project_id uuid references projects(id) on delete set null,
  dataset_id uuid references datasets(id) on delete set null,
  rq_job_id text unique,
  run_slug text,
  display_name text not null default 'cv_run',
  task_type text not null,
  model_type text not null,
  model_name text,
  params jsonb not null default '{}'::jsonb,
  status text not null default 'queued',
  storage_path text,
  latest_metrics jsonb,
  created_by text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  finished_at timestamptz,
  unique (workspace_id, run_slug)
);

alter table training_tasks alter column workspace_id drop not null;
alter table training_tasks alter column run_slug drop not null;
alter table training_tasks alter column storage_path drop not null;
alter table training_tasks add column if not exists display_name text;
alter table training_tasks add column if not exists owner_user_id text;
alter table training_tasks add column if not exists dataset_slug text;
alter table training_tasks add column if not exists error_detail text;
update training_tasks
set display_name = regexp_replace(run_slug, '_[0-9]{13}$', '')
where display_name is null and run_slug is not null;
update training_tasks set display_name = 'cv_run' where display_name is null;
alter table training_tasks alter column display_name set not null;
alter table training_tasks alter column display_name set default 'cv_run';

create table if not exists artifacts (
  id uuid primary key default gen_random_uuid(),
  training_run_id uuid not null references training_tasks(id) on delete cascade,
  kind text not null,
  file_name text not null,
  storage_path text not null,
  content_type text,
  size_bytes bigint,
  created_at timestamptz not null default now()
);

do $$
begin
  if exists (select 1 from pg_constraint where conname = 'training_runs_owner_user_fk')
     and not exists (select 1 from pg_constraint where conname = 'training_tasks_owner_user_fk') then
    alter table training_tasks rename constraint training_runs_owner_user_fk to training_tasks_owner_user_fk;
  end if;
  if to_regclass('idx_training_runs_workspace_id') is not null and to_regclass('idx_training_tasks_workspace_id') is null then
    alter index idx_training_runs_workspace_id rename to idx_training_tasks_workspace_id;
  end if;
  if to_regclass('idx_training_runs_created_by') is not null and to_regclass('idx_training_tasks_created_by') is null then
    alter index idx_training_runs_created_by rename to idx_training_tasks_created_by;
  end if;
  if to_regclass('idx_runs_owner_slug') is not null and to_regclass('idx_tasks_owner_slug') is null then
    alter index idx_runs_owner_slug rename to idx_tasks_owner_slug;
  end if;
  if to_regclass('idx_runs_dataset_status') is not null and to_regclass('idx_tasks_dataset_status') is null then
    alter index idx_runs_dataset_status rename to idx_tasks_dataset_status;
  end if;
end $$;

create index if not exists idx_workspace_members_user_id on workspace_members(user_id);
create index if not exists idx_projects_workspace_id on projects(workspace_id);
create index if not exists idx_datasets_workspace_id on datasets(workspace_id);
create index if not exists idx_training_tasks_workspace_id on training_tasks(workspace_id);
create index if not exists idx_training_tasks_created_by on training_tasks(created_by);
create unique index if not exists idx_datasets_owner_slug on datasets(owner_user_id, slug) where owner_user_id is not null;
create unique index if not exists idx_tasks_owner_slug on training_tasks(owner_user_id, run_slug) where owner_user_id is not null;
create index if not exists idx_tasks_dataset_status on training_tasks(dataset_id, status);

do $$
begin
  if to_regclass('"user"') is not null and not exists (select 1 from pg_constraint where conname = 'datasets_owner_user_fk') then
    alter table datasets add constraint datasets_owner_user_fk foreign key (owner_user_id)
      references "user"(id) on delete restrict not valid;
  end if;
  if to_regclass('"user"') is not null and not exists (select 1 from pg_constraint where conname = 'training_tasks_owner_user_fk') then
    alter table training_tasks add constraint training_tasks_owner_user_fk foreign key (owner_user_id)
      references "user"(id) on delete restrict not valid;
  end if;
end $$;

do $$
begin
  if exists (select 1 from pg_constraint where conname = 'datasets_owner_user_fk' and not convalidated) then
    alter table datasets validate constraint datasets_owner_user_fk;
  end if;
  if exists (select 1 from pg_constraint where conname = 'training_tasks_owner_user_fk' and not convalidated) then
    alter table training_tasks validate constraint training_tasks_owner_user_fk;
  end if;
end $$;
