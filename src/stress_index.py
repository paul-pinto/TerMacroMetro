from __future__ import annotations

from src.entities import analizar_contexto_boliviano


def main() -> None:
    ejemplos = [
        (
            "El BCB informó que las reservas internacionales continúan "
            "bajo presión debido a la escasez de divisas."
        ),
        (
            "YPFB anunció una mejora en la provisión de diésel y gasolina "
            "para el mercado interno."
        ),
        (
            "El INE reportó una caída de la inflación y una recuperación "
            "gradual del crecimiento económico."
        ),
    ]

    for indice, texto in enumerate(ejemplos, start=1):
        resultado = analizar_contexto_boliviano(texto)

        print(f"\nEJEMPLO {indice}")
        print(texto)
        print(resultado)


if __name__ == "__main__":
    main()
