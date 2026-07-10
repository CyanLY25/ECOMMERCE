-- Extensiones necesarias
create extension if not exists "pgcrypto";

-- Funcion para actualizar automaticamente el campo updated_at
create or replace function public.trigger_set_timestamp()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = timezone('utc', now());
  return new;
end;
$$;

-- Tabla de lineas (familias de productos)
create table if not exists public.lineas (
  id uuid primary key default gen_random_uuid(),
  nombre text not null,
  descripcion text,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create trigger trg_lineas_updated_at
before update on public.lineas
for each row
execute function public.trigger_set_timestamp();

-- Tabla de productos
create table if not exists public.productos (
  id uuid primary key default gen_random_uuid(),
  linea_id uuid references public.lineas(id) on delete set null,
  nombre text not null,
  sku text unique,
  stock numeric(12,2) not null default 0,
  stock_minimo numeric(12,2) not null default 0,
  price numeric(12,2) not null default 0,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create trigger trg_productos_updated_at
before update on public.productos
for each row
execute function public.trigger_set_timestamp();

-- Tabla de ordenes
create table if not exists public.ordenes (
  id uuid primary key default gen_random_uuid(),
  producto_id uuid references public.productos(id) on delete set null,
  cantidad numeric(12,2) not null,
  total numeric(14,2) not null default 0,
  estado text not null default 'pendiente',
  nota text,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create trigger trg_ordenes_updated_at
before update on public.ordenes
for each row
execute function public.trigger_set_timestamp();

-- Indices de apoyo
create index if not exists idx_productos_linea_id on public.productos (linea_id);
create index if not exists idx_productos_sku on public.productos (sku);
create index if not exists idx_ordenes_producto_id on public.ordenes (producto_id);
create index if not exists idx_ordenes_estado on public.ordenes (estado);
