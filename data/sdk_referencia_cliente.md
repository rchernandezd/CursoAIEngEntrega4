# techcorp-sdk — Referencia del cliente

Esta página documenta la clase `TCClient`, punto de entrada de la librería `tcsdk` para
consultar la API de Recursos Humanos, y su variante asincrónica. Todos los ejemplos
suponen que la configuración del entorno ya está hecha según la página de instalación y
configuración del paquete.

## Crear el cliente

La forma recomendada de construir un cliente es `TCClient.from_env()`, que lee el
token, el entorno y el timeout desde la configuración vigente, respetando la misma
precedencia que el resto de la librería. El cliente mantiene una sesión HTTP con
conexiones reutilizables, por lo que conviene usarlo como context manager: al salir del
bloque `with` se cierran las conexiones de forma ordenada.

```python
from tcsdk import TCClient

with TCClient.from_env() as client:
    print(client.env)           # "dev", "staging" o "prod"
```

También es posible crear el cliente sin context manager y llamar a `client.close()` al
terminar, pero es fácil olvidarlo y dejar conexiones abiertas en procesos largos.

## Consultar un empleado

El método `get_employee(employee_id)` devuelve un objeto `Employee` con los datos
públicos de la persona: identificador, nombre, área, líder directo y fecha de ingreso.
`Employee` es una dataclass inmutable (`frozen=True`), de modo que sus atributos no
pueden modificarse por accidente después de obtenerse; para trabajar con una versión
alterada se usa `dataclasses.replace`.

```python
from tcsdk import TCClient

with TCClient.from_env() as client:
    emp = client.get_employee("E-10234")
    print(emp.full_name, emp.area, emp.hire_date)
```

Si el identificador no existe, el método lanza una excepción de recurso no encontrado,
descrita en la página de errores de la librería.

## Listar solicitudes con paginación

El método `list_requests` devuelve las solicitudes de Recursos Humanos (vacaciones,
licencias, cambios de datos) filtradas por estado. Los resultados vienen paginados: el
parámetro `page_size` vale 50 por defecto y admite un máximo de 200 elementos por
página. Valores mayores son rechazados por la API con un error de validación.

Cada página expone sus elementos en `page.items` y el cursor de la página siguiente en
`page.next_cursor`. Para recorrer todos los resultados, se vuelve a llamar al método
pasando ese cursor, hasta que `next_cursor` vale `None`, lo que indica que no quedan más
páginas.

```python
from tcsdk import TCClient

with TCClient.from_env() as client:
    cursor = None
    while True:
        page = client.list_requests(status="pending", page_size=50, cursor=cursor)
        for req in page.items:
            print(req.id, req.kind, req.employee_id)
        cursor = page.next_cursor
        if cursor is None:
            break
```

Para procesos que solo necesitan iterar, el método auxiliar `client.iter_requests()`
encapsula este mismo bucle y devuelve los elementos uno por uno.

## Cliente asincrónico

`AsyncTCClient` expone exactamente la misma API que `TCClient`, pero cada método es una
corrutina que se invoca con `await`. Es la opción indicada para servicios web
asincrónicos o para consultar muchos empleados en paralelo sin abrir hilos adicionales.

```python
import asyncio

from tcsdk import AsyncTCClient


async def main():
    async with AsyncTCClient.from_env() as client:
        ids = ["E-10234", "E-10235", "E-10236"]
        empleados = await asyncio.gather(*(client.get_employee(i) for i in ids))
        for emp in empleados:
            print(emp.full_name)


asyncio.run(main())
```

No se deben mezclar llamadas sincrónicas y asincrónicas sobre el mismo cliente: cada
variante administra su propio conjunto de conexiones.

## Modelos de datos devueltos

Todos los objetos que devuelve el cliente son dataclasses inmutables definidas en el
módulo `tcsdk.models`, con anotaciones de tipos completas, lo que permite que los
editores y los verificadores estáticos detecten errores antes de ejecutar el código.
`Employee` incluye `employee_id`, `full_name`, `area`, `manager_id` y `hire_date`
(un objeto `date`). Cada solicitud de `list_requests` es un `HRRequest` con `id`,
`kind`, `status`, `employee_id` y `created_at`. Los valores posibles de `status` son
`pending`, `approved` y `rejected`; cualquier otro valor se rechaza antes de enviar la
solicitud. Para serializar un objeto a JSON se puede usar `dataclasses.asdict` y luego
convertir las fechas a texto en formato ISO-8601.
