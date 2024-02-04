import { Entity, PrimaryGeneratedColumn, Column } from "typeorm";
import { CreateImageDto } from "../dtos/CreateImage.dto";

@Entity()
export class Image {
  @PrimaryGeneratedColumn()
  imageId: number;

  @Column()
  foodImageUrl: string;

  mapCreateImageDtoToEntity(createImageDto: CreateImageDto) {
    const image = new Image();
    image.foodImageUrl = createImageDto.foodImageUrl;
    return image;
  }
}
