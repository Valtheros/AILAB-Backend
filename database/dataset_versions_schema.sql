create table if not exists dataset_collections (
  id uuid primary key default gen_random_uuid(),
  owner_user_id text not null constraint dataset_collections_owner_fk references "user"(id) on delete restrict,
  name text not null,
  label_studio_project_id uuid references label_studio_projects(id) on delete set null,
  reserved_version integer not null default 0 check (reserved_version >= 0),
  created_at timestamptz not null default now()
);

create index if not exists dataset_collections_owner on dataset_collections(owner_user_id, created_at desc);
create unique index if not exists dataset_collections_label_project
  on dataset_collections(label_studio_project_id) where label_studio_project_id is not null;

alter table datasets add column if not exists collection_id uuid references dataset_collections(id);
alter table datasets add column if not exists version integer;
alter table datasets add column if not exists content_sha256 text;
alter table datasets add column if not exists source_manifest_path text;
create unique index if not exists datasets_collection_version
  on datasets(collection_id, version) where collection_id is not null and version is not null;

create table if not exists dataset_import_intents (
  id uuid primary key default gen_random_uuid(),
  owner_user_id text not null constraint dataset_import_intents_owner_fk references "user"(id) on delete restrict,
  token_sha256 text not null,
  collection_id uuid not null references dataset_collections(id),
  version integer not null,
  slug text not null,
  metadata jsonb not null,
  content_sha256 text not null,
  dataset_id uuid references datasets(id) on delete set null,
  status text not null default 'pending' check (status in ('pending', 'completed')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique(owner_user_id, token_sha256),
  unique(collection_id, version)
);

do $$ begin
  if not exists (select 1 from pg_constraint where conname='dataset_collections_owner_fk' and conrelid='dataset_collections'::regclass) then
    alter table dataset_collections add constraint dataset_collections_owner_fk foreign key(owner_user_id) references "user"(id) on delete restrict;
  end if;
  if not exists (select 1 from pg_constraint where conname='dataset_import_intents_owner_fk' and conrelid='dataset_import_intents'::regclass) then
    alter table dataset_import_intents add constraint dataset_import_intents_owner_fk foreign key(owner_user_id) references "user"(id) on delete restrict;
  end if;
end $$;

alter table training_tasks add column if not exists dataset_content_sha256 text;
alter table training_tasks add column if not exists dataset_provenance text not null default 'unverified';
