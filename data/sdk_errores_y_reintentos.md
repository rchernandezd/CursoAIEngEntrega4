# techcorp-sdk — Errores y reintentos

Esta página describe las excepciones que lanza la librería `tcsdk` al comunicarse con
la API de Recursos Humanos, cuáles de ellas se reintentan de forma automática y cómo
ajustar la política de reintentos. Entender esta jerarquía permite escribir integraciones
robustas que distingan un problema de configuración de una falla pasajera del servicio.

## Jerarquía de excepciones

Todas las excepciones de la librería heredan de una clase base común, `TCError`, por lo
que capturar `TCError` alcanza para manejar cualquier error originado en el paquete. Las
subclases principales son:

- `TCAuthError`: la API respondió HTTP 401.
- `TCRateLimitError`: la API respondió HTTP 429.
- `TCServerError`: la API respondió con un error 5xx.
- `TCNotFoundError`: el recurso solicitado no existe (HTTP 404).
- `TCValidationError`: los parámetros enviados no son válidos (HTTP 422).

Cada excepción expone el código de estado en `exc.status` y el identificador de la
solicitud en `exc.request_id`, que conviene incluir en cualquier reporte al Equipo
Plataforma.

## Errores de autenticación

`TCAuthError` se lanza ante una respuesta HTTP 401, que indica que el token es inválido,
fue revocado o está vencido. Este error no se reintenta nunca, porque repetir la
solicitud con el mismo token produciría el mismo resultado. La solución es regenerar el
token desde el portal del Equipo Plataforma, actualizar la variable de entorno
correspondiente y reiniciar el proceso.

## Límite de solicitudes

La API aplica un límite de 600 solicitudes por minuto por token. Cuando se supera, la
respuesta es HTTP 429 y la librería lanza `TCRateLimitError`. La respuesta incluye el
header `Retry-After`, que indica cuántos segundos hay que esperar antes de volver a
intentar; el valor queda disponible en el atributo `exc.retry_after`. Respetar ese valor
es obligatorio: insistir antes de tiempo prolonga el bloqueo del token.

```python
import time

from tcsdk import TCClient, TCRateLimitError

with TCClient.from_env() as client:
    try:
        emp = client.get_employee("E-10234")
    except TCRateLimitError as exc:
        # La API indica en Retry-After cuántos segundos esperar.
        time.sleep(exc.retry_after)
        emp = client.get_employee("E-10234")
```

Para procesos masivos se recomienda distribuir las llamadas en el tiempo en lugar de
depender de los reintentos. El límite se cuenta por token y no por máquina, de modo que
varios procesos que comparten un mismo token de servicio consumen el mismo cupo; en
esos casos conviene usar un token distinto por proceso o coordinar las llamadas con una
cola central. El portal del Equipo Plataforma muestra el consumo de cada token en los
últimos minutos, lo que ayuda a dimensionar la carga antes de lanzar un proceso nuevo.

## Errores del servidor

`TCServerError` agrupa las respuestas 5xx, que suelen deberse a problemas transitorios
del servicio, como despliegues o picos de carga. A diferencia de los errores de
autenticación y validación, estos errores son reintentables y la librería los reintenta
automáticamente según la política configurada.

## Política de reintentos

Los reintentos automáticos se controlan con la clase `RetryPolicy`. La política por
defecto es `RetryPolicy(max_attempts=3, backoff_factor=0.5)`, es decir, 3 intentos en
total con backoff exponencial: la espera entre intentos se calcula como `backoff_factor`
multiplicado por dos elevado al número de intento, más una pequeña variación aleatoria
para evitar que muchos clientes reintenten a la vez. Se reintentan `TCServerError`,
`TCRateLimitError` (respetando `Retry-After`) y los timeouts de red.

```python
from tcsdk import RetryPolicy, TCClient

# Más paciencia para un proceso nocturno que no tiene apuro.
policy = RetryPolicy(max_attempts=5, backoff_factor=1.0)

with TCClient.from_env(retry_policy=policy) as client:
    emp = client.get_employee("E-10234")
```

Con `max_attempts=1` se desactivan los reintentos, lo que resulta útil en pruebas
automatizadas que necesitan fallar rápido.

## Buenas prácticas

Capturar las excepciones más específicas primero y `TCError` al final, registrar siempre
el `request_id` en los logs y no ocultar los errores de autenticación detrás de
reintentos genéricos. Si un error persiste después de agotar los reintentos, conviene
abrir un ticket al Equipo Plataforma con el identificador de la solicitud.
